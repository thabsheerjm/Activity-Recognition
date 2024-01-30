#!/usr/bin/env python3

def get_unique_elements(list_):
    unique = []
    for element in list_:
        if element not in unique:
            unique.append(element)
    return unique

    