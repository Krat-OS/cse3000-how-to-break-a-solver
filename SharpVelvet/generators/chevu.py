#!/usr/bin/env python3

import sys
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

class BipartiteGraph:
    """
    Represents a bipartite graph with clauses and variables as nodes.
    The graph is used to generate constrained CNF formulas.
    """
    def __init__(self, num_clauses=0, num_vars=0, edges=None):
        self.num_clauses = num_clauses
        self.num_vars = num_vars
        self.edges = edges if edges is not None else []

    @classmethod
    def create_graph(cls,
                     min_clauses=80, max_clauses=120,
                     min_vars=40, max_vars=70,
                     min_clause_len=2, max_clause_len=8,
                     min_refs=4, max_refs=8,
                     allow_taut=False,
                     balanced=False,
                     clause_to_var_ratio=None):
        """
        Create a bipartite graph representing a CNF formula subject to constraints:
        - The number of clauses and variables can be defined or constrained.
        - Clauses are generated with lengths and references based on specified bounds.
        - Balancing ensures more even variable usage.
        - Allows or disallows tautologies in clauses.

        Returns:
            BipartiteGraph: A bipartite graph object representing the CNF formula.
        """
        num_vars = random.randint(min_vars, max_vars)

        # Determine the number of clauses based on the ratio or fixed range
        if clause_to_var_ratio is not None:
            num_clauses = int(clause_to_var_ratio * num_vars)
        else:
            num_clauses = random.randint(min_clauses, max_clauses)

        edges = []  # List to store edges (clause-literal pairs)
        references = {v: 0 for v in range(1, num_vars + 1)}  # Track variable references

        def pick_variable():
            """
            Picks a variable for inclusion in a clause. Balancing ensures less-used
            variables are selected first if enabled.

            Returns:
                int or None: The selected variable or None if no valid variable exists.
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

        previous_clause_literals = None

        # Generate each clause
        for c in range(1, num_clauses + 1):
            chosen_literals = set()

            if c % 2 == 1:
                # Odd-numbered clauses: Randomly generated
                clause_len = min_clause_len + int(
                    (max_clause_len - min_clause_len) * (random.random() ** 4)
                )
                for _ in range(clause_len):
                    var = pick_variable()
                    if var is None:
                        break

                    sign = random.choice([True, False])
                    literal = var if sign else -var

                    if not allow_taut and -literal in chosen_literals:
                        continue

                    if literal in chosen_literals:
                        continue

                    chosen_literals.add(literal)
                    references[abs(literal)] += 1
            else:
                # Even-numbered clauses: Constrained generation
                if previous_clause_literals:
                    chosen_literals = set(previous_clause_literals)
                    flip_count = random.randint(1, len(chosen_literals))
                    literals_to_flip = random.sample(list(chosen_literals), flip_count)

                    for lit in literals_to_flip:
                        chosen_literals.remove(lit)
                        references[abs(lit)] -= 1
                        flipped_lit = -lit

                        if not allow_taut and flipped_lit in chosen_literals:
                            chosen_literals.add(lit)
                            references[abs(lit)] += 1
                            continue

                        if flipped_lit in chosen_literals:
                            chosen_literals.add(lit)
                            references[abs(lit)] += 1
                            continue

                        chosen_literals.add(flipped_lit)
                        references[abs(flipped_lit)] += 1

                    num_literals_to_replace = random.choice([0, len(chosen_literals) - 1])
                    
                    for _ in range(num_literals_to_replace):
                        if chosen_literals:
                            lit_to_replace = random.choice(list(chosen_literals))
                            chosen_literals.remove(lit_to_replace)
                            references[abs(lit_to_replace)] -= 1
                            var = pick_variable()
                            
                            if var is not None:
                                sign = random.choice([True, False])
                                new_lit = var if sign else -var
                                
                                if not allow_taut and -new_lit in chosen_literals:
                                    chosen_literals.add(lit_to_replace)
                                    references[abs(lit_to_replace)] += 1
                                elif new_lit in chosen_literals:
                                    chosen_literals.add(lit_to_replace)
                                    references[abs(lit_to_replace)] += 1
                                else:
                                    chosen_literals.add(new_lit)
                                    references[abs(new_lit)] += 1

            previous_clause_literals = chosen_literals.copy()
            for lit in chosen_literals:
                edges.append((c, lit))

        # Ensure each variable meets the minimum reference count
        for var in range(1, num_vars + 1):
            while references[var] < min_refs:
                clause_lengths = {clause_idx: 0 for clause_idx in range(1, num_clauses + 1)}
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

                edges.append((c, literal))
                references[var] += 1

        return cls(num_clauses, num_vars, edges)

    def to_cnf_string(self):
        """
        Convert the bipartite graph to CNF DIMACS format as a string.

        Returns:
            str: The CNF DIMACS formatted string.
        """
        num_clauses = self.num_clauses
        num_vars = self.num_vars

        clause_dict = {c: [] for c in range(1, num_clauses + 1)}
        for clause, literal in self.edges:
            clause_dict[clause].append(literal)

        lines = [f"p cnf {num_vars} {num_clauses}", "c t mc"]
        for c in range(1, num_clauses + 1):
            clause_line = " ".join(map(str, clause_dict[c])) + " 0"
            lines.append(clause_line)

        return "\n".join(lines) + "\n"

    def ensure_satisfiable(self, solution=None):
        """
        Ensure the formula is satisfiable by modifying literals to satisfy each clause.

        Args:
            solution (dict): A dictionary mapping variable IDs to their Boolean values.
                             If None, a random solution is generated.

        Returns:
            dict: The solution ensuring the formula's satisfiability.
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


def generate_instance(index, args):
    """
    Generate a single bipartite graph instance and ensure satisfiability.

    Args:
        index (int): The instance index.
        args (Namespace): Parsed command-line arguments.

    Returns:
        str: The CNF DIMACS formatted string for the instance.
    """
    if args.seed is not None:
        random.seed(args.seed + index)

    graph = BipartiteGraph.create_graph(
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

    return graph.to_cnf_string()

def main():
    """
    Main function for generating CNF instances using parallel threads.
    """
    parser = argparse.ArgumentParser(
        description="Generate bipartite graphs (CNF formulas) with constraints in parallel."
    )
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Set the random seed (default: None).")
    parser.add_argument("--instances", type=int, default=1,
                        help="Number of instances to generate (default: 1).")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of threads to use for parallel generation (default: 4).")

    parser.add_argument("--min-clauses", type=int, default=500,
                        help="Minimum number of clauses (default=500).")
    parser.add_argument("--max-clauses", type=int, default=500,
                        help="Maximum number of clauses (default=500).")
    parser.add_argument("--min-vars", type=int, default=140,
                        help="Minimum number of variables (default=140).")
    parser.add_argument("--max-vars", type=int, default=140,
                        help="Maximum number of variables (default=140).")
    parser.add_argument("--min-clause-len", type=int, default=3,
                        help="Minimum clause length (default=3).")
    parser.add_argument("--max-clause-len", type=int, default=3,
                        help="Maximum clause length (default=3).")
    parser.add_argument("--min-refs", type=int, default=1,
                        help="Minimum references per variable (default=1).")
    parser.add_argument("--max-refs", type=int, default=50,
                        help="Maximum references per variable (default=30).")

    parser.add_argument("--allow-taut", action="store_true",
                        help="Allow tautological clauses (default=False).")
    parser.add_argument("--balanced", action="store_true",
                        help="Make distribution of variables more balanced (default=False).")

    parser.add_argument("--ratio", type=float, default=None,
                        help="Clause-to-variable ratio (overrides min/max clauses if given).")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(generate_instance, i, args) for i in range(args.instances)]
        for future in futures:
            cnf_string = future.result()
            if cnf_string is not None:
                sys.stdout.write(cnf_string)

if __name__ == "__main__":
    main()
