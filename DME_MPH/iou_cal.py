import shapely.geometry
import shapely.affinity

class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

class BBoxIOU:
    def __init__(self, data):
        assert len(data) == 4
        self.xmin, self.xmax, self.ymin, self.ymax = data
    def for_iou(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)
    
    def get_rotated_box(self) -> RotatedRect:
        cx = (self.xmin + self.xmax)*0.5
        cy = (self.ymin + self.ymax)*0.5
        w = self.xmax - self.xmin
        h = self.ymax - self.ymin
        angle = 0
        return RotatedRect(cx, cy, w, h, angle)
    

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # 0,1,2,3 => xmin, ymin, xmax, ymax,

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_iou(box1, box2):
    assert type(box1) == BBoxIOU
    assert type(box2) == BBoxIOU
    return bb_intersection_over_union(box1.for_iou(), box2.for_iou())




if __name__ == '__main__':
    r1 = RotatedRect(10, 15, 15, 10, 30)
    r2 = RotatedRect(15, 15, 20, 10, 0)
    a = r1.intersection(r2).area
    print('success', a)

