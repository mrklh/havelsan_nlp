def merge_overlaps(ranges):
    ranges_sorted = sorted(ranges, key=lambda tup: tup[0])
    merged = []

    for next_el in ranges_sorted:
        if merged:
            last_el = merged[-1]
            if next_el[0] <= last_el[1]:
                if last_el[2] == next_el[2]:
                    max_val = max(last_el[1], next_el[1])
                    merged[-1] = (last_el[0], max_val, last_el[2])
                else:
                    if last_el[1] <= next_el[1]:
                        merged[-1] = (last_el[0], next_el[1], next_el[2])
                    else:
                        merged[-1] = last_el
            else:
                merged.append(next_el)
        else:
            merged.append(next_el)

    return merged
