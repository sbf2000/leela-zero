/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors
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

#ifndef UCTNODE_H_INCLUDED
#define UCTNODE_H_INCLUDED

#include "config.h"

#include <atomic>
#include <memory>
#include <vector>
#include <cassert>
#include <cstring>

#include "GameState.h"
#include "Network.h"
#include "SMP.h"
#include "UCTNodePointer.h"
#include "UCTSearch.h"

class UCTNode {
public:
    // When we visit a node, add this amount of virtual losses
    // to it to encourage other CPUs to explore other parts of the
    // search tree.
    static constexpr auto VIRTUAL_LOSS_COUNT = 3;
    // Defined in UCTNode.cpp
    explicit UCTNode(int vertex, float score);
    UCTNode() = delete;
    ~UCTNode() = default;

    bool create_children(std::atomic<int>& nodecount,
                         GameState& state, float& value, float& alpkt,
			 float& beta,
                         float min_psa_ratio = 0.0f);

    const std::vector<UCTNodePointer>& get_children() const;
    void sort_children(int color);
    void sort_children_by_policy();
    UCTNode& get_best_root_child(int color);
    UCTNode* uct_select_child(const GameState & currstate, bool is_root,
                              int max_visits,
                              std::vector<int> move_list,
                              bool nopass = false);

    size_t count_nodes() const;
    SMP::Mutex& get_mutex();
    bool first_visit() const;
    bool has_children() const;
    bool expandable(const float min_psa_ratio = 0.0f) const;
    void invalidate();
    void set_active(const bool active);
    bool valid() const;
    bool active() const;
    double get_blackevals() const;
    int get_move() const;
    int get_visits() const;
    float get_score() const;
    void set_score(float score);
    float get_eval(int tomove) const;
    float get_net_eval(int tomove) const;
    float get_agent_eval(int tomove) const;
    float get_eval_bonus() const;
    float get_eval_bonus_father() const;
    void set_eval_bonus_father(float bonus);
    float get_eval_base() const;
    float get_eval_base_father() const;
    void set_eval_base_father(float bonus);
    float get_net_eval() const;
    float get_net_beta() const;
    float get_net_alpkt() const;
    void set_values(float value, float alpkt, float beta);
#ifndef NDEBUG
    void set_urgency(float urgency, float psa, float q,
                     float num, float den);
    std::array<float, 5> get_urgency() const;
#endif
    void virtual_loss(void);
    void virtual_loss_undo(void);
    void clear_visits(void);
    void clear_children_visits(void);
    void update(float eval);

    // Defined in UCTNodeRoot.cpp, only to be called on m_root in UCTSearch
    bool randomize_first_proportionally();
    void prepare_root_node(int color,
                           std::atomic<int>& nodecount,
                           GameState& state,
                           bool fast_roll_out = false);

    UCTNode* get_first_child() const;
    UCTNode* get_second_child() const;
    UCTNode* get_nopass_child(FastState& state) const;
    std::unique_ptr<UCTNode> find_child(const int move);
    void inflate_all_children();
    UCTNode* select_child(int move);
    float estimate_alpkt(int passes, bool is_tromptaylor_scoring = false) const;

private:
    enum Status : char {
        INVALID, // superko
        PRUNED,
        ACTIVE
    };
    void link_nodelist(std::atomic<int>& nodecount,
                       std::vector<Network::ScoreVertexPair>& nodelist,
                       float min_psa_ratio);
    void accumulate_eval(float eval);
    void kill_superkos(const KoState& state);
    void dirichlet_noise(float epsilon, float alpha);
    void get_subtree_alpkts(std::vector<float> & vector, int passes,
                            bool is_tromptaylor_scoring) const;

    // Note : This class is very size-sensitive as we are going to create
    // tens of millions of instances of these.  Please put extra caution
    // if you want to add/remove/reorder any variables here.

    // Move
    std::int16_t m_move;
    // UCT
    std::atomic<std::int16_t> m_virtual_loss{0};
    std::atomic<int> m_visits{0};
    // UCT eval
    float m_score;
    // Original net eval for this node (not children).
    float m_net_eval{0.5f};
    //    float m_net_value{0.5f};
    float m_net_alpkt{0.0f}; // alpha + \tilde k
    float m_net_beta{1.0f};
    float m_eval_bonus{0.0f}; // x bar
    float m_eval_base{0.0f}; // x base
    float m_eval_base_father{0.0f}; // x base of father node
    float m_eval_bonus_father{0.0f}; // x bar of father node
#ifndef NDEBUG
    std::array<float, 5> m_last_urgency;
#endif

    // the following is used only in fpu, with reduction
    float m_agent_eval{0.5f}; // eval_with_bonus(eval_bonus()) no father
    std::atomic<double> m_blackevals{0.0};
    std::atomic<Status> m_status{ACTIVE};
    // Is someone adding scores to this node?
    bool m_is_expanding{false};
    SMP::Mutex m_nodemutex;

    // Tree data
    std::atomic<float> m_min_psa_ratio_children{2.0f};
    std::vector<UCTNodePointer> m_children;
};

#endif
