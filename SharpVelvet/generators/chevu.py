#!/usr/bin/env python3

import sys
import random
import argparse

class BipartiteGraph:
    def __init__(self, num_clauses=0, num_vars=0, edges=None):
        self.num_clauses = num_clauses
        self.num_vars = num_vars
        self.edges = edges if edges is not None else []

    @classmethod
    def create_random(cls, num_clauses, num_vars, max_edges=None):
        """
        Original random generator (unchanged).
        Creates a random bipartite graph with given number of clauses and variables.
        Clause nodes: 1..num_clauses
        Var nodes:    1..num_vars (positive) and -1..-num_vars (negative)
        """
        if max_edges is None or max_edges > num_clauses * (2 * num_vars):
            max_edges = num_clauses * (2 * num_vars)

        num_edges = random.randint(0, max_edges)
        clause_nodes = list(range(1, num_clauses + 1))
        var_nodes = list(range(1, num_vars + 1)) + list(range(-1, -num_vars - 1, -1))

        all_possible_edges = [(c, v) for c in clause_nodes for v in var_nodes]
        chosen_edges = random.sample(all_possible_edges, num_edges)

        return cls(num_clauses, num_vars, chosen_edges)

    @classmethod
    def create_constrained_random(cls,
                                  min_clauses=80, max_clauses=120,
                                  min_vars=40, max_vars=70,
                                  min_clause_len=2, max_clause_len=8,
                                  min_refs=4, max_refs=8,
                                  allow_taut=False,
                                  balanced=False,
                                  clause_to_var_ratio=None):
        """
        Create a bipartite graph subject to constraints:
          - #clauses in [min_clauses, max_clauses] OR determined by clause_to_var_ratio
          - #vars in [min_vars, max_vars]
          - Each clause length in [min_clause_len, max_clause_len] (skewed toward smaller)
          - Each variable has at least min_refs and at most max_refs references
          - If balanced=True, distribute variable usage more evenly
          - If allow_taut=False, do not allow x and -x in the same clause
          - If clause_to_var_ratio is not None, override #clauses = int(ratio * num_vars)
        """
        num_vars = random.randint(min_vars, max_vars)

        if clause_to_var_ratio is not None:
            num_clauses = int(clause_to_var_ratio * num_vars)
        else:
            num_clauses = random.randint(min_clauses, max_clauses)

        edges = []
        references = {v: 0 for v in range(1, num_vars + 1)}

        def pick_variable():
            """
            If balanced=True, pick from the least-used variables
            Otherwise pick any variable that hasn't hit max_refs.
            """
            can_use = [v for v in range(1, num_vars + 1) if references[v] < max_refs]
            if not can_use:
                return None

            if balanced:
                min_usage = min(references[v] for v in can_use)
                least_used = [v for v in can_use if references[v] == min_usage]
                return random.choice(least_used)
            else:
                return random.choice(can_use)

        for c in range(1, num_clauses + 1):
            clause_len = min_clause_len + int((max_clause_len - min_clause_len) * (random.random()**4))
            chosen_literals = set()

            for _ in range(clause_len):
                var = pick_variable()
                if var is None:
                    break

                sign = random.choice([True, False]) 
                literal = var if sign else -var

                if literal in chosen_literals:
                    continue
                if not allow_taut and -literal in chosen_literals:
                    continue

                references[var] += 1
                chosen_literals.add(literal)

            for lit in chosen_literals:
                edges.append((c, lit))

        for var in range(1, num_vars + 1):
            while references[var] < min_refs:
                clause_lengths = {
                    clause_idx: 0 for clause_idx in range(1, num_clauses + 1)
                }
                for clause_idx, lit in edges:
                    clause_lengths[clause_idx] += 1

                available_clauses = [
                    c for c, length in clause_lengths.items() if length < max_clause_len
                ]
                if not available_clauses:
                    break

                c = random.choice(available_clauses)
                sign = random.choice([True, False])
                literal = var if sign else -var

                existing_literals = [lit for cl, lit in edges if cl == c]
                if not allow_taut and -literal in existing_literals:
                    continue
                if literal in existing_literals:
                    continue

                references[var] += 1
                edges.append((c, literal))

        return cls(num_clauses, num_vars, edges)

    def _format_clause(self, c):
        """Format a clause node for text output."""
        return f"C{c}"

    def _format_var(self, v):
        """Format a variable node for text output."""
        if v >= 0:
            return f"X{v}"
        else:
            return f"NX{-v}"

    def to_file(self, filename):
        """
        Save the graph to a text file in the specified format:
        Clauses: N
        Variables: M
        Edges:
        C1 X1
        C1 NX2
        ...
        """
        with open(filename, 'w') as f:
            f.write(f"Clauses: {self.num_clauses}\n")
            f.write(f"Variables: {self.num_vars}\n")
            f.write("Edges:\n")

            sorted_edges = sorted(self.edges, key=lambda x: (x[0], x[1]))

            for c, v in sorted_edges:
                f.write(f"{self._format_clause(c)} {self._format_var(v)}\n")

    @classmethod
    def from_file(cls, filename):
        """
        Load a graph from a text file in the specified format.
        """
        with open(filename, 'r') as f:
            line = f.readline().strip()
            _, n_str = line.split(':')
            num_clauses = int(n_str.strip())

            line = f.readline().strip()
            _, m_str = line.split(':')
            num_vars = int(m_str.strip())

            line = f.readline().strip()
            if not line.startswith("Edges"):
                raise ValueError("Expected 'Edges:' line not found.")

            edges = []
            for line in f:
                if line.strip():
                    c_str, v_str = line.strip().split()
                    c = int(c_str[1:])
                    if v_str.startswith("NX"):
                        v = -int(v_str[2:])
                    else:
                        v = int(v_str[1:])
                    edges.append((c, v))

            return cls(num_clauses, num_vars, edges)

    def __str__(self):
        """
        Return a string representation of the bipartite graph in the same
        format as the file:
          Clauses: N
          Variables: M
          Edges:
          C1 X1
          C2 NX2
          ...
        """
        lines = [
            f"Clauses: {self.num_clauses}",
            f"Variables: {self.num_vars}",
            "Edges:"
        ]

        sorted_edges = sorted(self.edges, key=lambda x: (x[0], x[1]))
        for c, v in sorted_edges:
            lines.append(f"{self._format_clause(c)} {self._format_var(v)}")
        return "\n".join(lines)

    def to_cnf_string(self):
        """
        Return the CNF DIMACS format as a string (no intermediate file).
        Format:
          - Header: p cnf {num_vars} {num_clauses}
          - Comment: c t mc
          - Clauses: each clause as a line ending with 0
        """
        num_clauses = self.num_clauses
        num_vars = self.num_vars

        clause_dict = {c: [] for c in range(1, num_clauses + 1)}
        for clause, literal in self.edges:
            clause_dict[clause].append(literal)

        lines = []
        lines.append(f"p cnf {num_vars} {num_clauses}")
        lines.append("c t mc")
        for c in range(1, num_clauses + 1):
            clause_line = " ".join(map(str, clause_dict[c])) + " 0"
            lines.append(clause_line)

        return "\n".join(lines) + "\n"

    def to_cnf_file(self, filename="instance.cnf"):
        """
        Converts the bipartite graph to CNF DIMACS format and saves it to a file.
        (Kept for compatibility, but we won't use it in the main loop.)
        """
        cnf_text = self.to_cnf_string()
        with open(filename, "w") as f:
            f.write(cnf_text)

    def ensure_satisfiable(self, solution=None):
        """
        Modifies edges so that the formula is satisfied by the given solution.
        If solution is None, a random solution is created (var->bool).
        Then for each clause that is unsatisfied by that solution,
        one literal is flipped (if needed) to make that clause satisfied.

        Returns:
            dict: The solution (var -> bool) that satisfies the formula.
        """
        if solution is None:
            solution = {v: random.choice([True, False]) for v in range(1, self.num_vars + 1)}

        clause_dict = {}
        for c, lit in self.edges:
            clause_dict.setdefault(c, []).append(lit)

        for c in clause_dict:
            lits = clause_dict[c]
            clause_satisfied = False
            for lit in lits:
                var_id = abs(lit)
                sign = (lit > 0)
                if solution[var_id] == sign:
                    clause_satisfied = True
                    break

            if not clause_satisfied and lits:
                idx_to_flip = random.randrange(len(lits))
                old_lit = lits[idx_to_flip]
                var_id = abs(old_lit)

                needed_sign = solution[var_id]
                new_lit = var_id if needed_sign else -var_id

                lits[idx_to_flip] = new_lit

        new_edges = []
        for c in clause_dict:
            for lit in clause_dict[c]:
                new_edges.append((c, lit))
        self.edges = new_edges

        return solution


###############################################################################
# Main script entry point for SharpVelvet compatibility + generating instances
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate bipartite graphs (CNF formulas) with constraints."
    )
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Set the random seed (default: None).")
    parser.add_argument("--instances", type=int, default=25,
                        help="Number of instances to generate (default: 25).")

    parser.add_argument("dummy", nargs="?", help="Dummy argument for compatibility.", default=None)

    parser.add_argument("--min-clauses", type=int, default=250,
                        help="Minimum number of clauses (default=250).")
    parser.add_argument("--max-clauses", type=int, default=260,
                        help="Maximum number of clauses (default=260).")
    parser.add_argument("--min-vars", type=int, default=60,
                        help="Minimum number of variables (default=60).")
    parser.add_argument("--max-vars", type=int, default=120,
                        help="Maximum number of variables (default=120).")
    parser.add_argument("--min-clause-len", type=int, default=3,
                        help="Minimum clause length (default=3).")
    parser.add_argument("--max-clause-len", type=int, default=8,
                        help="Maximum clause length (default=8).")
    parser.add_argument("--min-refs", type=int, default=3,
                        help="Minimum references per variable (default=3).")
    parser.add_argument("--max-refs", type=int, default=15,
                        help="Maximum references per variable (default=15).")

    parser.add_argument("--allow-taut", action="store_true",
                        help="Allow tautological clauses (default=False).")
    parser.add_argument("--balanced", action="store_true",
                        help="Make distribution of variables more balanced (default=False).")

    parser.add_argument("--ratio", type=float, default=None,
                        help="Clause-to-variable ratio (overrides min/max clauses if given).")

    args = parser.parse_args()

    for i in range(1, args.instances + 1):
        graph = BipartiteGraph.create_constrained_random(
            min_clauses=args.min_clauses,
            max_clauses=args.max_clauses,
            min_vars=args.min_vars,
            max_vars=args.max_vars,
            min_clause_len=args.min_clause_len,
            max_clause_len=args.max_clause_len,
            min_refs=args.min_refs,
            max_refs=args.max_refs,
            allow_taut=args.allow_taut,
            balanced=args.balanced,
            clause_to_var_ratio=args.ratio
        )

        solution = {v: random.choice([True, False]) for v in range(1, graph.num_vars + 1)}
        graph.ensure_satisfiable(solution=solution)

    if graph.to_cnf_string() is not None:
        sys.stdout.write(graph.to_cnf_string())
