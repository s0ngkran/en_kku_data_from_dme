import os

# from dme_mph import Point
class Point:
    def __init__(self, data=None, x=None, y=None):
        if x is None or y is None:
            assert type(data) == list
            assert len(data) == 3
            self.x, self.y, self.z = data
        else:
            self.x, self.y = x, y


class RotatedRect:
    def __init__(self, data):
        # data = 0.45166666666666666,0.45166666666666666;0.4796752626552054,0.5524976122254059;0.35967526265520533,0.5858309455587393;0.33166666666666667,0.485;
        def get_box(data):
            box = []
            points = data.split(';')[:-1]
            for point in points:
                point = point.split(',')
                x, y = point
                p = Point(x=x, y=y)
                box.append(p)
            return box
        self.points = get_box(data)

def test():
    data_path = './rotated_rect/12.txt'
    with open(data_path, 'r') as f:
        dat = f.read()
    rect = RotatedRect(dat)
    for point in rect.points:
        print(point.x, point.y)

if __name__ == '__main__':
    test()