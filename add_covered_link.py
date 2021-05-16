import json
import copy

# load json
with open('validation_json', 'r') as f:
    data = json.load(f)

def gen_covered_link_map():
    link25 =  [[0,1],[1,2],[2,3],[0,4],[5,6],[6,7],[7,8],[4,9],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],[21,22],[22,23],[23,24]]
    # create covered link map <covered_keypoint, covered_link>
    covered_link_map = {}
    for covered_link_index, (a, b) in enumerate(link25):
        covered_link_index = str(covered_link_index)
        a, b = str(a), str(b)
        if a not in covered_link_map:
            covered_link_map.update({
                a:covered_link_index,
            })
        else:
            covered_link_map[a] += ','+covered_link_index
        if b not in covered_link_map:
            covered_link_map.update({
                b:covered_link_index,
            })
        else:
            covered_link_map[b] += ','+covered_link_index
    return covered_link_map

    
for cnt, dat in enumerate(data):
    print(cnt, len(data))
    keypoint = dat['keypoint']

    # get covered_index
    covered_index = []
    covered_point = []
    for index, (x,y, covered) in enumerate(keypoint):
        covered = True if covered=='1' else False
        covered_point.append(covered)
        if covered == '1':
            covered_index.append(index)

    # gen covered link
    covered_link_map = gen_covered_link_map()
    covered_link = {}
    for index in covered_index:
        index = str(index)
        update_list = covered_link_map[index].split(',')
        for ind in update_list:
            covered_link.update({
                ind:'covered',
            })

    ans = [False for i in range(23*2)]   
    for i in covered_link.keys():
        ans[int(i)*2] = True
        ans[int(i)*2+1] = True

    covered_link = ans

    # update data
    dat['covered_link'] = copy.copy(covered_link)
    dat['covered_point'] = copy.copy(covered_point)

# update json
with open('validation_json_with_covered', 'w') as f:
    json.dump(data, f)

    

