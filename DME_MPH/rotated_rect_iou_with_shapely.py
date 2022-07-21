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



if __name__ == '__main__':
    r1 = RotatedRect(10, 15, 15, 10, 30)
    r2 = RotatedRect(15, 15, 20, 10, 0)
    a = r1.intersection(r2).area
    print('success', a)
