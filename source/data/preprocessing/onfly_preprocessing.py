import copy
import random
from typing import List, Tuple, Optional
from random import uniform
import warnings
import numpy as np
from copy import deepcopy

def pretty_print(list) :
    print("[")
    for elt in list :
        if elt[0] != 8 :
            print(elt)
    print("]")

def random_masking(mask : int,
                  input_ids : List[int],
                  spans : List[Tuple[int, int]],
                  random_proba : float) -> List[int] :
    """
    Apply random masking over entities defined by a list of spans
    :param input_ids:
    :param spans:
    :param random_proba:
    :return:
    """

    if mask is not None :
        for span in spans :
            if uniform(0, 1) < random_proba :
                if span[1] != span[0] :
                    input_ids[span[0] : span[1]] = [mask] * (span[1] - span[0])
                else :
                    input_ids[span[0]] = mask
    else :
        warnings.warn("### WARNING : Unknown token id has not been set, random "
                      "masking strategy cannot be used. Continuing without masking.")
    return input_ids




def swap_players(sample , threshold : float = 1.0) :
    """
    Swap player name within a text without changing the labels associated
    :param example: Training sample
    :return: Updated training sample
    """

    def look_for_entity(buffer : List, entities : List) -> Optional[str] :

        # getting token span
        begin = buffer[0][0]
        end = buffer[-1][1]

        for entity in entities :
            if 'token_span' in entity :
                token_span = entity['token_span']
                if begin == token_span[0] and end == token_span[1] + 1 :
                    return entity['name']
            else :
                continue

        return None

    def create_segment(span_list, weights, labels, entities, ids, masks) :
        res = []
        buffer = []
        label_ref = labels[0]
        weight_ref = weights[0]
        mask_ref = span_list[0][2]
        for index, span in enumerate(span_list) :

            if len(span) == 3 :
                label = 0
            else :
                print(span)
                label = span[3] + 1

            mask = span[2]

            if label_ref != label or mask_ref != mask  :

                segment = {
                    "segment" : buffer,
                    "label" : label_ref,
                    "weight" : weights[index],
                    "ids" : ids[buffer[0][0] : buffer[-1][1]],
                    "mask" : mask_ref
                }

                if label_ref != 0 :

                    print(f"check for : {buffer}")
                    entity_name = look_for_entity(buffer, entities)

                    if entity_name is None :
                        print("###########")
                        print(res)
                        raise RuntimeError(f"No entity found in list for span {buffer}")

                    segment['name'] = entity_name

                res.append(segment)
                label_ref = label
                weight_ref = weight_ref
                mask_ref = mask
                buffer = [span]
                # if last element register segment
                if index == (len(span_list) - 1) :

                    segment = {
                          "segment" : buffer,
                          "label" : label,
                          "weight" : weights[buffer[0][0]],
                          "ids" : ids[buffer[0][0] : buffer[-1][1]],
                          "mask" : mask
                      }

                    res.append(segment)
            else :
                buffer.append(span)
        return res

    def  create_entity_list(segment_list) -> Tuple[List, List] :
        entity_list = []
        names = set()
        for segment in segment_list :
            if "name" in segment and segment['name'] not in names :
                entity_list.append(segment)
                names.add(segment['name'])

        random.Random().shuffle(entity_list)

        len_list = len(entity_list) // 2

        list_1 = entity_list[0:len_list]
        list_2 = entity_list[len_list:2]

        return list_1, list_2

    def swap_entity(entity_1, entity_2, segment_list) -> List :
        e_1 = copy.deepcopy(entity_1)
        e_2 = copy.deepcopy(entity_2)
        e_1['label'], e_2['label'] = e_2['label'], e_1['label']
        e_1['weight'], e_2['weight'] = e_2['weight'], e_1['weight']

        res = []
        for segment in segment_list :
            # non entity segment are left as it
            if "name" not in segment :
                res.append(segment)
            else :
                # swapping segments
                if segment['name'] == e_1['name'] :
                    print("#############SWAP")
                    res.append(e_2)
                elif segment['name'] == e_2['name'] :
                    print("#############SWAP")

                    res.append(e_1)
                else :
                    # no swapping if it is not a relevent entity
                    res.append(segment)

        return res

    def generate_sample(segment_list) :
        gt = []
        mask_gt = []
        weights = []
        ids = []
        span_list = []
        index_span = 0
        attention_mask = []

        for segment in segment_list :
            for span in segment['segment'] :
                gt.append(segment['label'])
                weights.append(segment['weight'])
                mask_gt.append(segment['mask'])
                begin = index_span
                end = index_span + span[1] - span[0]
                if segment['label'] != 0 :
                    span_list.append((begin, end, segment['mask'], segment['label']))
                else :
                    span_list.append((begin, end, segment['mask']))

                index_span = end

            ids += segment['ids']

            attention_mask += [segment['mask']] * len(segment['ids'])

        return gt, mask_gt, weights, ids, span_list, attention_mask

    span_list = sample.span_list
    entities = sample.mentions
    input_ids = sample.input_ids
    weights = sample.weights_sequence
    labels = sample.labels_sequence
    mask_label = sample.label_mask

    segment_list = create_segment(span_list, weights, labels, entities, input_ids, mask_label)

    '''
    print(f"segmet list : \n")
    for seg in segment_list :
        print("===")
        for k, v in seg.items() :
            print(f"{k} : {v}")
    '''
    
    entity_list_1, entity_list_2 = create_entity_list(segment_list)
    print(f"@####### SWAP TA MERE {entity_list_1} {entity_list_2} ")
    for entity_1, entity_2 in zip(entity_list_1, entity_list_2) :
        if True : #uniform(0, 1) < threshold :
            print("swap")
            segment_list = swap_entity(entity_1, entity_2, segment_list)

    '''
    print(f"SGMENT SWAPPED list : \n")
    for seg in segment_list:
        print("===")
        for k, v in seg.items():
            print(f"{k} : {v}")
    '''
    labels_sequence, label_mask, weights_sequence, input_ids,  span_list, attention_mask = generate_sample(segment_list)

    sample.labels_sequence = labels_sequence
    sample.label_mask = label_mask
    sample.weights_sequence = weights_sequence
    sample.input_ids = input_ids
    sample.span_list = span_list
    sample.attention_mask = attention_mask

    return sample

onfly_preprocessing = {
    "task_1_random_masking" : random_masking,
    "task_2_entity_swaping" :  swap_players
}
