from typing import Dict, List, Optional, Sequence, Tuple, TypeVar, Union
from rdkit import Chem
from fuseprop import find_clusters, extract_subgraph, get_mol, get_smiles, find_fragments
from copy import deepcopy
import numpy as np
import torch
from deg import *
from agent import sample


T = TypeVar('T')


def data_processing(input_smiles: List[str], GNN_model_path: str, motif=False) -> Tuple[SubGraphSet, Dict[MolKey, InputGraph]]:
    input_mols = []
    input_graphs = []
    init_subgraphs = []
    subgraphs_idx = []
    input_graphs_dict = {}
    init_edge_flag = 0

    for n, smiles in enumerate(input_smiles):
        print("data processing {}/{}".format(n, len(input_smiles)))
        # Kekulized
        smiles = get_smiles(get_mol(smiles))
        mol = get_mol(smiles)
        input_mols.append(mol)
        if not motif:
            clusters, atom_cls = find_clusters(mol)
            for i,cls in enumerate(clusters):
                clusters[i] = set(list(cls))
            for a in range(len(atom_cls)):
                atom_cls[a] = set(atom_cls[a])
        else:
            fragments = find_fragments(mol)
            clusters = [frag[1] for frag in fragments]
        
        # Construct graphs
        subgraphs = []
        subgraphs_idx_i = [] 
        for i,cluster in enumerate(clusters):
            _, subgraph_i_mapped, _ = extract_subgraph(smiles, cluster)
            subgraphs.append(SubGraph(subgraph_i_mapped, mapping_to_input_mol=subgraph_i_mapped, subfrags=list(cluster)))
            subgraphs_idx_i.append(list(cluster))
            init_edge_flag += 1
        
        init_subgraphs.append(subgraphs)
        subgraphs_idx.append(subgraphs_idx_i)
        graph = InputGraph(mol, smiles, subgraphs, subgraphs_idx_i, GNN_model_path)
        input_graphs.append(graph)
        input_graphs_dict[MolKey(graph.mol)] = graph

    # Construct subgraph_set
    subgraph_set = SubGraphSet(init_subgraphs, subgraphs_idx, input_graphs)
    return subgraph_set, input_graphs_dict


def grammar_generation(
    agent: torch.nn.Module, 
    input_graphs_dict: Dict[MolKey, InputGraph], 
    subgraph_set: SubGraphSet, 
    grammar: ProductionRuleCorpus, 
    mcmc_iter: int, 
    sample_number: int, 
    args
) -> Tuple[bool, Dict, SubGraphSet, ProductionRuleCorpus]:
    # Selected hyperedge (subgraph)
    plist = [*subgraph_set.map_to_input]

    # Terminating condition
    if len(plist) == 0:
        # done_flag, new_input_graphs_dict, new_subgraph_set, new_grammar
        return True, input_graphs_dict, subgraph_set, grammar

    # Update every InputGraph: remove every subgraph that equals to p_star, 
    # for those subgraphs that contain atom idx in p_star, replace the 
    # atom with p_star
    org_input_graphs_dict = deepcopy(input_graphs_dict)
    org_subgraph_set = deepcopy(subgraph_set)
    org_grammar = deepcopy(grammar)

    input_graphs_dict = deepcopy(org_input_graphs_dict)
    subgraph_set = deepcopy(org_subgraph_set)
    grammar = deepcopy(org_grammar)

    # for i, (key, input_g) in enumerate(input_graphs_dict.items()):
    for i, input_g in enumerate(input_graphs_dict.values()):
        print("---for graph {}---".format(i))
        action_list = []
        all_final_features = []
        # Skip the final iteration for training agent
        if len(input_g.subgraphs) > 1:
            for subgraph, subgraph_idx in zip(input_g.subgraphs, input_g.subgraphs_idx):
                subg_feature = input_g.get_subg_feature_for_agent(subgraph)
                num_occurance = subgraph_set.map_to_input[MolKey(subgraph)][1]
                num_in_input = len(subgraph_set.map_to_input[MolKey(subgraph)][0].keys())
                final_feature = []
                final_feature.extend(subg_feature.tolist())
                final_feature.append(1 - np.exp(-num_occurance))
                final_feature.append(num_in_input / len(list(input_graphs_dict.keys())))
                all_final_features.append(
                    torch.unsqueeze(torch.from_numpy(np.array(final_feature)).float(), 
                                    0)
                )
            
            while True:
                action_list, take_action = sample(
                    agent, 
                    torch.vstack(all_final_features), 
                    mcmc_iter, 
                    sample_number
                )
                if take_action:
                    break
        elif len(input_g.subgraphs) == 1:
            action_list = [1]
        else:
            continue

        print("Hyperedge sampling:", action_list)
        # Merge connected hyperedges
        p_star_list = input_g.merge_selected_subgraphs(action_list)
        # Generate rules
        for p_star in p_star_list:
            is_inside, subgraphs, subgraphs_idx = input_g.is_candidate_subgraph(p_star)
            if is_inside:
                for subg, subg_idx in zip(subgraphs, subgraphs_idx):
                    if subg_idx not in input_g.subgraphs_idx:
                        # Skip the subg if it has been merged in previous iterations
                        continue
                    grammar = generate_rule(input_g, subg, grammar)
                    input_g.update_subgraph(subg_idx)
                    
    # Update subgraph_set
    subgraph_set.update([g for (k, g) in input_graphs_dict.items()])
    new_grammar = deepcopy(grammar)
    new_input_graphs_dict = deepcopy(input_graphs_dict)
    new_subgraph_set = deepcopy(subgraph_set)
    return False, new_input_graphs_dict, new_subgraph_set, new_grammar


def MCMC_sampling(
    agent: torch.nn.Module, 
    all_input_graphs_dict: Dict, 
    all_subgraph_set: SubGraphSet, 
    all_grammar: ProductionRuleCorpus, 
    sample_number: int, 
    args
) -> Tuple[int, ProductionRuleCorpus, Dict]:
    """Markov Chain Monte Carlo sampling for grammar generation."""

    iter_num = 0
    while True:
        print("======MCMC iter{}======".format(iter_num))
        
        done_flag, new_input_graphs_dict, new_subgraph_set, new_grammar = grammar_generation(
            agent, 
            all_input_graphs_dict, 
            all_subgraph_set, 
            all_grammar, 
            iter_num, 
            sample_number, 
            args
        )

        print("Graph contraction status: ", done_flag)
        if done_flag:
            break
        
        all_input_graphs_dict = deepcopy(new_input_graphs_dict)
        all_subgraph_set = deepcopy(new_subgraph_set)
        all_grammar = deepcopy(new_grammar)
        
        iter_num += 1

    return iter_num, new_grammar, new_input_graphs_dict



def random_produce(
    grammar: ProductionRuleCorpus,
    *,
    max_steps: int = 30
) -> Union[Tuple[Chem.RWMol, int], Tuple[None, int]]:
    """Returns a random molecule sampled from the grammar (and the number of steps).
    
    Returns `None` if the process fails to produce a molecule.
    """
    
    def sample(l: Sequence[T], prob: Optional[Sequence[float]]=None) -> Tuple[T, int]:
        if prob is None:
            prob = [1/len(l)] * len(l)
        idx =  np.random.choice(range(len(l)), 1, p=prob)[0]
        return l[idx], idx

    def prob_schedule(step: int, selected_idx: List[int]):
        """
        prob = exp(a * t * x), x = {0, 1}
            where x indicates if the current rule is an 
            ending rule
        
        NOTE: This make is less likely for ending rules, 
            and impossible for starting rules
        """
        
        a = 0.5

        prob_list = [
            0 if rule.is_start_rule
            else np.exp(a * step * rule.is_ending)
            for rule in grammar.prod_rule_list
        ]

        # mask selected idx
        prob_list = np.array(prob_list)[selected_idx]
        
        # normalize
        prob_list = prob_list / np.sum(prob_list)
        
        return prob_list

    hypergraph = Hypergraph()
    starting_rules = [
        rule 
        for rule in grammar.prod_rule_list 
        if rule.is_start_rule
    ]

    step = 0
    while True:
        if step == 0:
            # NOTE: samples a random starting rule and applies it to the hg
            selected_rule, idx = sample(starting_rules)
            candidate_hg, _, available = selected_rule.graph_rule_applied_to(hypergraph)
            hypergraph = deepcopy(candidate_hg)
        else:
            # NOTE: tries to apply all the production rules in the grammar, and keeps track of the ones that work
            candidate_rules: List[ProductionRule] = []
            candidate_rules_idx: List[int] = []
            candidate_hgs: List[Hypergraph] = []
            for rule_i, rule in enumerate(grammar.prod_rule_list):
                candidate_hg, _, available = rule.graph_rule_applied_to(hypergraph)
                if available:
                    candidate_rules.append(rule)
                    candidate_rules_idx.append(rule_i)
                    candidate_hgs.append(candidate_hg)
            
            # NOTE: If all if the candidate rules are starting rules or the number of steps is larger than `max_steps`, stop
            if (all(rule.is_start_rule for rule in candidate_rules) and step > 0) or step > max_steps:
                break
            
            hypergraph, idx = sample(
                candidate_hgs, 
                prob_schedule(step, candidate_rules_idx)
            )
            selected_rule = candidate_rules[idx]
        
        step += 1

    try:
        mol = hg_to_mol(hypergraph)
        print(Chem.MolToSmiles(mol))
    except:
        return None, step

    return mol, step
