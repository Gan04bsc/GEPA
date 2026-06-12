def does_triplet_have_desirable_predictors(program_candidates, ancestor, id1, id2):
    found_predictors = []
    for pred_idx, (pred_anc, pred_id1, pred_id2) in enumerate(zip(
        program_candidates[ancestor].named_predictors(),
        program_candidates[id1].named_predictors(),
        program_candidates[id2].named_predictors()
    )):
        if (
            (pred_anc[1].signature.instructions == pred_id1[1].signature.instructions) or 
            (pred_anc[1].signature.instructions == pred_id2[1].signature.instructions)
        ) and (
            pred_id1[1].signature.instructions != pred_id2[1].signature.instructions
        ):
            # We have a predictor that is the same as one of its ancestors, so we can update it with the other
            same_as_ancestor_id = (1 if pred_anc[1].signature.instructions == pred_id1[1].signature.instructions else 2)
            found_predictors.append((pred_idx, same_as_ancestor_id))
    
    return len(found_predictors) > 0

def filter_ancestors(i, j, common_ancestors, merges_performed, agg_scores, program_candidates):
    filtered_ancestors = []
    for ancestor in common_ancestors:
        if (i, j, ancestor) in merges_performed[0]:
            continue

        if agg_scores[ancestor] > agg_scores[i] or agg_scores[ancestor] > agg_scores[j]:
            continue

        if not does_triplet_have_desirable_predictors(program_candidates, ancestor, i, j):
            continue

        filtered_ancestors.append(ancestor)
    return filtered_ancestors

def find_common_ancestor_pair(rng, parent_list, program_indexes, merges_performed, agg_scores, program_candidates, max_attempts=10):
    def get_ancestors(node, ancestors_found):
        parents = parent_list[node]
        for parent in parents:
            if parent is not None and parent not in ancestors_found:
                ancestors_found.add(parent)
                get_ancestors(parent, ancestors_found)
        
        return list(ancestors_found)

    for _ in range(max_attempts):
        if len(program_indexes) < 2:
            return None
        i, j = rng.sample(program_indexes, 2)
        if i == j:
            continue

        if j < i:
            i, j = j, i

        ancestors_i = get_ancestors(i, set())
        ancestors_j = get_ancestors(j, set())

        if j in ancestors_i or i in ancestors_j:
            # If one is an ancestor of the other, we cannot merge them
            continue

        common_ancestors = set(ancestors_i) & set(ancestors_j)
        common_ancestors = filter_ancestors(i, j, common_ancestors, merges_performed, agg_scores, program_candidates)
        if common_ancestors:
            # Select a random common ancestor
            common_ancestor = rng.choices(common_ancestors, k=1, weights=[agg_scores[ancestor] for ancestor in common_ancestors])[0]
            return (i, j, common_ancestor)

    return None

def sample_and_attempt_merge_programs_by_common_predictors(agg_scores, rng, merge_candidates, merges_performed, program_candidates, parent_program_for_candidate, max_attempts=10):
    if len(merge_candidates) < 2:
        return (False, None, None, None, None)
    if len(parent_program_for_candidate) < 3:
        return (False, None, None, None, None)

    for _ in range(max_attempts):
        ids_to_merge = find_common_ancestor_pair(rng, parent_program_for_candidate, list(merge_candidates), merges_performed=merges_performed, agg_scores=agg_scores, program_candidates=program_candidates, max_attempts=10)
        if ids_to_merge is None:
            continue
        id1, id2, ancestor = ids_to_merge

        assert (id1, id2, ancestor) not in merges_performed, "This pair has already been merged"

        assert agg_scores[ancestor] <= agg_scores[id1], "Ancestor should not be better than its descendants"
        assert agg_scores[ancestor] <= agg_scores[id2], "Ancestor should not be better than its descendants"
        assert id1 != id2, "Cannot merge the same program"

        # Now we have a common ancestor, which is outperformed by both its descendants

        new_program = program_candidates[ancestor].deepcopy()
        curr_prog_lm = program_candidates[ancestor].get_lm()
        new_program.set_lm(curr_prog_lm)

        new_prog_desc = ()

        for pred_idx, (pred_anc, pred_id1, pred_id2) in enumerate(zip(
            program_candidates[ancestor].named_predictors(),
            program_candidates[id1].named_predictors(),
            program_candidates[id2].named_predictors()
        )):
            if (
                (pred_anc[1].signature.instructions == pred_id1[1].signature.instructions) or 
                (pred_anc[1].signature.instructions == pred_id2[1].signature.instructions)
            ) and (
                pred_id1[1].signature.instructions != pred_id2[1].signature.instructions
            ):
                # We have a predictor that is the same as one of its ancestors, so we can update it with the other
                same_as_ancestor_id = (1 if pred_anc[1].signature.instructions == pred_id1[1].signature.instructions else 2)
                new_program.named_predictors()[pred_idx][1].signature = program_candidates[id2 if same_as_ancestor_id == 1 else id1].named_predictors()[pred_idx][1].signature
                new_prog_desc = (*new_prog_desc, id2 if same_as_ancestor_id == 1 else id1)
            elif (
                (pred_anc[1].signature.instructions != pred_id1[1].signature.instructions) and
                (pred_anc[1].signature.instructions != pred_id2[1].signature.instructions)
            ):
                # Both predictors are different from  the ancestor, and it is difficult to decide which one gives the benefits
                # We randomly select one of the descendants to update the predictor
                # The probability of selecting is proportional to the agg_scores of the descendants
                # prog_to_get_instruction_from = id1 if (rng.random() < (agg_scores[id1] / (agg_scores[id1] + agg_scores[id2]))) else id2
                prog_to_get_instruction_from = id1 if (agg_scores[id1] > agg_scores[id2]) else (id2 if agg_scores[id2] > agg_scores[id1] else rng.choice([id1, id2]))
                new_program.named_predictors()[pred_idx][1].signature = program_candidates[prog_to_get_instruction_from].named_predictors()[pred_idx][1].signature
                new_prog_desc = (*new_prog_desc, prog_to_get_instruction_from)
            elif (
                pred_id1[1].signature.instructions == pred_id2[1].signature.instructions
            ):
                # Either both predictors are the same, or both are different from the ancestor
                # If both are different from the ancestor, we should use the new predictor, so selecting either one of the descendants is fine
                # If both are same as the ancesor, again selecting any one of the descendants is fine
                # So let's select id1
                new_program.named_predictors()[pred_idx][1].signature = program_candidates[id1].named_predictors()[pred_idx][1].signature
                new_prog_desc = (*new_prog_desc, id1)
            else:
                assert False, "Unexpected case in predictor merging logic"
            
        if (id1, id2, new_prog_desc) in merges_performed[1]:
            # This triplet has already been merged, so we skip it
            continue
            
        merges_performed[1].append((id1, id2, new_prog_desc))

        return (True, new_program, id1, id2, ancestor)
    
    return (False, None, None, None, None)
