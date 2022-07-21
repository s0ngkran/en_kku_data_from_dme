import random
class Side:
    def __init__(self, name, objs):
        self.name = name
        self.objs = objs
        self.first_time = True
    def move_to(self, other_side, n):

        # select obj to move
        random.shuffle(self.objs)
        if self.first_time and self.name=='right-side':
            self.first_time = False
            res = True
            moving = random.choice([[-8,8],[5,-5],[3,-3]])
         
            for i in moving:
                obj = self.objs.remove(i)
                other_side.receive(i)
            action = self.cvt_moving_to_text(moving, other_side)
           
            return res, action
            

        moving = []
        if len(self.objs) <= 2:
            n = 1
        # find person
        for i in self.objs:
            if i < 0:
                self.objs.remove(i)
                moving.append(i)
                other_side.receive(i)
                break
        if n == 2:
            obj = self.objs.pop(0)
            moving.append(obj)
            other_side.receive(obj)

        action = self.cvt_moving_to_text(moving, other_side)
        if self.is_valid(moving) and other_side.is_valid(moving):
            res = True
        else: 
            res = False
        print(res, self.objs, other_side.objs)
        return res, action
    def cvt_moving_to_text(self, moving, other_side):
        text = 'move'
        for i in moving:
            text += ' ' + str(i)
        text += ' to ' + other_side.name +'\n'
        return text
    def receive(self, obj):
        self.objs.append(obj)
    def is_valid(self, moving):
        my_map = {
            'red':-8,
            'green':-5,
            'blue':-3,
            '8':8,
            '5':5,
            '3':3
        }

        found_person = False
        for i in self.objs:
            if i < 0:
                found_person = True
        
        if found_person:
            sum_ = sum(self.objs)
            if sum_ > 0:
                return False

        return True
def run():
    left = Side('left-side', [])
    right = Side('right-side', [-8,-5,-3,8,5,3])
    cnt = 0
    moving_success = []
    while True:
        success, action = right.move_to(left, 2)
        if not success: 
            break
        moving_success.append(action)
        cnt += 1

        # check win
        print(len(left.objs), 'len left objs')
        if len(left.objs) == 6:
            return cnt, moving_success
        success, action = left.move_to(right, random.randint(1,2))
        if not success:
            break
        moving_success.append(action)
        cnt += 1
    return 9999, moving_success

print('del old file')
with open('./boat_result.txt', 'w') as f:
    f.write('')
solution = []
r = 0
while True:
    cnt, action = run()
    if cnt != 9999:
        if action not in solution:
            print(cnt, r+1, end="")
            r += 1
            solution.append(action)
            print('found solution', r)
            with open('./boat_result.txt', 'a') as f:
                f.write('solution ' + str(r)+'\n')
                for context in action:
                    f.write(context)
                f.write('-----------\n')
            break
            if r == 2000:
                break
        




        
