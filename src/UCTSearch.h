/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto
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

#ifndef UCTSEARCH_H_INCLUDED
#define UCTSEARCH_H_INCLUDED

#include <list>
#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <future>

#include "ThreadPool.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GameState.h"
#include "UCTNode.h"
#include "Utils.h"


class SearchResult {
public:
    SearchResult() = default;
    bool valid() const { return m_valid;  }
    float eval() const { return m_value;  }
    float eval_with_bonus(float bonus, float base);
    static SearchResult from_eval(float value, float alpkt, float beta) {
        return SearchResult(value, alpkt, beta);
    }
    static SearchResult from_score(float board_score) {
        return SearchResult(Utils::winner(board_score), board_score, 10.0f);
    }
private:
    explicit SearchResult(float value, float alpkt, float beta)
        : m_valid(true), m_value(value), m_alpkt(alpkt), m_beta(beta) {}
    bool m_valid{false};
    float m_value{0.5f};
    float m_alpkt{0.0f};
    float m_beta{1.0f};
};

namespace TimeManagement {
    enum enabled_t {
        AUTO = -1, OFF = 0, ON = 1, FAST = 2
    };
};

class UCTSearch {
public:
    /*
        Depending on rule set and state of the game, we might
        prefer to pass, or we might prefer not to pass unless
        it's the last resort. Same for resigning.
    */
    using passflag_t = int;
    static constexpr passflag_t NORMAL   = 0;
    static constexpr passflag_t NOPASS   = 1 << 0;
    static constexpr passflag_t NORESIGN = 1 << 1;

    /*
        Maximum size of the tree in memory. Nodes are about
        48 bytes, so limit to ~1.2G on 32-bits and about 5.5G
        on 64-bits.
    */
    static constexpr auto MAX_TREE_SIZE =
        (sizeof(void*) == 4 ? 25'000'000 : 100'000'000);

    /*
        Value representing unlimited visits or playouts. Due to
        concurrent updates while multithreading, we need some
        headroom within the native type.
    */
    static constexpr auto UNLIMITED_PLAYOUTS =
        std::numeric_limits<int>::max() / 2;

    static constexpr auto FAST_ROLL_OUT_VISITS = 20;
    static constexpr auto EXPLORE_MOVE_VISITS = 30;

    UCTSearch(GameState& g);
    int think(int color, passflag_t passflag = NORMAL);
    void set_playout_limit(int playouts);
    void set_visit_limit(int visits);
    void ponder();
    bool is_running() const;
    void increment_playouts();
    SearchResult play_simulation(GameState& currstate,
                                 UCTNode* const node);
    float final_japscore();
    
private:
    float get_min_psa_ratio() const;
    void dump_stats(FastState& state, UCTNode& parent);
    void print_move_choices_by_policy(KoState& state, UCTNode& parent,
                                      int at_least_as_many, float probab_threash);
    void tree_stats(const UCTNode& node);
    std::string get_pv(FastState& state, UCTNode& parent);
    void dump_analysis(int playouts);
    bool should_resign(passflag_t passflag, float bestscore);
    bool have_alternate_moves(int elapsed_centis, int time_for_move);
    int est_playouts_left(int elapsed_centis, int time_for_move) const;
    size_t prune_noncontenders(int elapsed_centis = 0, int time_for_move = 0);
    bool stop_thinking(int elapsed_centis = 0, int time_for_move = 0) const;
    int get_best_move(passflag_t passflag);
    void update_root();
    bool advance_to_new_rootstate();
    void select_playable_dame(FullBoard *board);
    void select_dame_sequence(FullBoard *board);
    bool is_stopping (int move) const;
    bool is_better_move(int move1, int move2, float & estimated_score);
    void explore_move(int move);
    void explore_root_nopass();
    void fast_roll_out();

    GameState & m_rootstate;
    std::unique_ptr<GameState> m_last_rootstate;
    std::unique_ptr<UCTNode> m_root;
    std::atomic<int> m_nodes{0};
    std::atomic<int> m_playouts{0};
    std::atomic<bool> m_run{false};
    int m_maxplayouts;
    int m_maxvisits;

    // Advanced search parameters
    bool m_chn_scoring = true;
    
    // Max number of visits per node: nodes with this or more visits
    // are never selected. Acts on first generation children of root
    // node, since the deeper generations always have fewer visits.
    // If equal to 0 it is ignored.
    int m_per_node_maxvisits = 0;

    // List of moves allowed as first generation choices during the
    // search. Only applies to the first move in the simulation.
    // If empty it is ignored.
    std::vector<int> m_allowed_root_children = {};

    // If, during the search, any of these vertexes is the move of a
    // node with at least m_stopping_visits, the flag is set to
    // true.  If the vector is empty or the visits are 0 it is
    // ignored.
    std::vector<int> m_stopping_moves = {};
    int m_stopping_visits = 0;
    bool m_stopping_flag = false;
    bool m_nopass = false;
    
    int m_bestmove = FastBoard::PASS;

    std::list<Utils::ThreadGroup> m_delete_futures;
};

class UCTWorker {
public:
    UCTWorker(GameState & state, UCTSearch * search, UCTNode * root)
      : m_rootstate(state), m_search(search), m_root(root) {}
    void operator()();
private:
    GameState & m_rootstate;
    UCTSearch * m_search;
    UCTNode * m_root;
};

#endif
