/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Gian-Carlo Pascutto
    Copyright (C) 2018 SAI Team

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "config.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "FastBoard.h"
#include "FastState.h"
#include "KoState.h"
#include "Random.h"
#include "UCTNode.h"
#include "Utils.h"
#include "GTP.h"
#include "Network.h"

/*
 * These functions belong to UCTNode but should only be called on the root node
 * of UCTSearch and have been seperated to increase code clarity.
 */

using Utils::myprintf;

UCTNode* UCTNode::get_first_child() const {
    if (m_children.empty()) {
        return nullptr;
    }

    m_children.front().inflate();
    return m_children.front().get();
}

UCTNode* UCTNode::get_second_child() const {
    if (m_children.size() < 2) {
        return nullptr;
    }

    m_children[1].inflate();
    return m_children[1].get();
}

void UCTNode::kill_superkos(const KoState& state) {
    for (auto& child : m_children) {
        auto move = child->get_move();
        if (move != FastBoard::PASS) {
            KoState mystate = state;
            mystate.play_move(move);

            if (mystate.superko()) {
                // Don't delete nodes for now, just mark them invalid.
                child->invalidate();
            }
        }
    }

    // Now do the actual deletion.
    m_children.erase(
        std::remove_if(begin(m_children), end(m_children),
                       [](const auto &child) { return !child->valid(); }),
        end(m_children)
    );
}

void UCTNode::dirichlet_noise(float epsilon, float alpha) {
    auto child_cnt = m_children.size();

    auto dirichlet_vector = std::vector<float>{};
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    for (size_t i = 0; i < child_cnt; i++) {
        dirichlet_vector.emplace_back(gamma(Random::get_Rng()));
    }

    auto sample_sum = std::accumulate(begin(dirichlet_vector),
                                      end(dirichlet_vector), 0.0f);

    // If the noise vector sums to 0 or a denormal, then don't try to
    // normalize.
    if (sample_sum < std::numeric_limits<float>::min()) {
        return;
    }

    for (auto& v : dirichlet_vector) {
        v /= sample_sum;
    }

    child_cnt = 0;
    for (auto& child : m_children) {
        auto score = child->get_score();
        auto eta_a = dirichlet_vector[child_cnt++];
        score = score * (1 - epsilon) + epsilon * eta_a;
        child->set_score(score);
    }
}

bool UCTNode::randomize_first_proportionally() {
    auto accum = 0.0;
    auto norm_factor = 0.0;
    auto accum_vector = std::vector<double>{};
    auto prb_vector = std::vector<float>{};


    for (const auto& child : m_children) {
        auto visits = child->get_visits();

        if (norm_factor == 0.0) {
            norm_factor = visits;
            // Nonsensical options? End of game?
            if (visits <= cfg_random_min_visits) {
                return false;
            }
        }
        if (visits > cfg_random_min_visits) {
            accum += std::pow(visits / norm_factor,
                              1.0 / cfg_random_temp);
            accum_vector.emplace_back(accum);
	    prb_vector.emplace_back(visits);
        }
    }

    auto distribution = std::uniform_real_distribution<double>{0.0, accum};
    auto pick = distribution(Random::get_Rng());
    auto index = size_t{0};
    for (size_t i = 0; i < accum_vector.size(); i++) {
        if (pick < accum_vector[i]) {
            index = i;
            break;
        }
    }

#ifndef NDEBUG
    myprintf("Rnd_first: accum=%f, pick=%f, index=%d.\n", accum, pick, index);
#endif

    // Take the early out
    if (index == 0) {
        return false;
    }

    assert(m_children.size() > index);

    // Now swap the child at index with the first child
    std::iter_swap(begin(m_children), begin(m_children) + index);

    const bool is_dumb_move = (prb_vector[index] / prb_vector[0] < cfg_blunder_thr);

#ifndef NDEBUG
    myprintf("Randomize_first: p=%f over p0=%f, move is %s\n",
	     prb_vector[index], prb_vector[0], (is_dumb_move ? "blunder" : "ok") );
#endif

    return is_dumb_move;
}

UCTNode* UCTNode::get_nopass_child(FastState& state) const {
    for (const auto& child : m_children) {
        /* If we prevent the engine from passing, we must bail out when
           we only have unreasonable moves to pick, like filling eyes.
           Note that this knowledge isn't required by the engine,
           we require it because we're overruling its moves. */
        if (child->m_move != FastBoard::PASS
            && !state.board.is_eye(state.get_to_move(), child->m_move)) {
            return child.get();
        }
    }
    return nullptr;
}

// Used to find new root in UCTSearch.
std::unique_ptr<UCTNode> UCTNode::find_child(const int move) {
    for (auto& child : m_children) {
        if (child.get_move() == move) {
             // no guarantee that this is a non-inflated node
            child.inflate();
            return std::unique_ptr<UCTNode>(child.release());
        }
    }

    // Can happen if we resigned or children are not expanded
    return nullptr;
}

void UCTNode::inflate_all_children() {
    for (const auto& node : get_children()) {
        node.inflate();
    }
}

void UCTNode::prepare_root_node(int color,
                                std::atomic<int>& nodes,
                                GameState& root_state,
                                bool fast_roll_out) {
    float root_value, root_alpkt, root_beta;

    const auto had_children = has_children();
    if (expandable()) {
        create_children(nodes, root_state, root_value, root_alpkt, root_beta);
    }
    if (has_children() && !had_children) {
	// blackevals is useless here because root nodes are never
	// evaluated, nevertheless the number of visits must be updated
	update(0);
    }
    
    //    root_eval = get_net_eval(color);
    //    root_eval = (color == FastBoard::BLACK ? root_eval : 1.0f - root_eval);

#ifndef NDEBUG
    myprintf("NN eval=%f. Agent eval=%f\n", get_net_eval(color), get_agent_eval(color));
#else
    if (!fast_roll_out) {
        myprintf("NN eval=%f. Agent eval=%f\n", get_net_eval(color), get_agent_eval(color));
    }
#endif

    // There are a lot of special cases where code assumes
    // all children of the root are inflated, so do that.
    inflate_all_children();

    // Remove illegal moves, so the root move list is correct.
    // This also removes a lot of special cases.
    kill_superkos(root_state);

    if (fast_roll_out) {
        return;
    }
    
    if (cfg_noise) {
        // Adjust the Dirichlet noise's alpha constant to the board size
        auto alpha = cfg_noise_value * 361.0f / BOARD_SQUARES;
        dirichlet_noise(0.25f, alpha);
    }

    if (cfg_japanese_mode) {
        for (auto& child : m_children) {
            auto score = child->get_score();
            score *= 0.8f;
            if (child->get_move() == FastBoard::PASS) {
                score += 0.2f;
            }
            child->set_score(score);
        }
    }
}

