import numpy as np
import matplotlib.pyplot as plt
def is_point_in_rotated_box(shape=None, point=None, test=False, plot=False):
    if plot == True:
        assert test == False
    def is_on_right_side(x, y, xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        a = float(y1 - y0)
        b = float(x0 - x1)
        c = - a*x0 - b*y0
        return a*x + b*y + c >= 0

    def test_point(x, y, vertices):
        num_vert = len(vertices)
        is_right = [is_on_right_side(x, y, vertices[i], vertices[(i + 1) % num_vert]) for i in range(num_vert)]
        all_left = not any(is_right)
        all_right = all(is_right)
        return all_left or all_right

    vertices = [(670273, 4879507), (677241, 4859302), (670388, 4856938), (663420, 4877144)]
    vertices1 = [(670273, 4879507), (677241, 4859302), (670388, 4856938), (663420, 4877144)]
    vertices2 = [(680000, 4872000), (680000, 4879000), (690000, 4879000), (690000, 4872000)]
    vertices3 = [(655000, 4857000), (655000, 4875000), (665000, 4857000)]
    k = np.arange(6)
    r = 8000
    vertices4 = np.vstack([690000 + r * np.cos(k * 2 * np.pi / 6), 4863000 + r * np.sin(k * 2 * np.pi / 6)]).T
    # all_shapes = [vertices1, vertices2, vertices3, vertices4]
    all_shapes = [vertices]

    # for vertices in all_shapes:
    #     plt.plot([x for x, y in vertices] + [vertices[0][0]], [y for x, y in vertices] + [vertices[0][1]], 'g-', lw=3)
    if plot:
        for x, y in shape:
            plt.plot(x, y, 'or')
        x, y = point.x, point.y
        plt.plot(x, y, 'og')
        shape[3], shape[2] = shape[2], shape[3]
        ans = test_point(x, y, shape)
        x, y = 200, 75
        ans2 = test_point(x, y, shape)
        plt.plot(x, y, 'oc')
        plt.title(str(ans) + str(ans2))
        plt.show()
    if test:
        for x, y in zip(np.random.randint(650000, 700000, 1000), np.random.randint(4855000, 4880000, 1000)):
            if not test:
                x, y = point.x, point.y

            color = 'turquoise'
            for vertices in all_shapes:
                if test_point(x, y, vertices):
                    color = 'tomato'
            if test: 
                plt.plot(x, y, '.', color=color)
                # plt.title('point ', str(color))
            if not test:
                return True if color == 'tomato' else False
        plt.show()
    else:
        return test_point(point.x, point.y, shape)

    # plt.gca().set_aspect('equal')

def main():
    is_point_in_rotated_box(test=True)

if __name__ == '__main__':
    main()