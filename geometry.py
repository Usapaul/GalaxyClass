
import numpy as np


#-----------------------------------------------


class Point2D(tuple):
    ''' Just a regular point in 2D-space which has x and y coordinates
        input: two values — X and Y coordinates
    '''
    #
    def __new__(cls, x, y):
        ''' Point2D == tuple (x,y) '''
        instance = super().__new__(cls, (x,y))
        return instance

    def __init__(self, x, y):
        if int(x) != x or int(y) != y:
            x = float(x)
            y = float(y)
        else:
            x = int(x)
            y = int(y)
        self.x = x
        self.y = y
        #
        self._radial_distance = np.sqrt(x**2 + y**2)
        # calculate positional angle (argphi) counter clockwise
        # from 0 to 360 degrees
        if x == 0:
            self._PA = 90 * np.sign(y)
        elif y == 0:
            self._PA = 0 if x >= 0 else 180
        else:
            angle = np.degrees(np.arctan(y / x))
            if x > 0:
                if y >= 0:
                    self._PA = angle
                else:
                    self._PA = angle + 360
            else:
                self._PA = angle + 180


    @classmethod
    def from_tuple(cls, input_tuple):
        x, y = input_tuple
        return cls(x, y)

    @classmethod
    def from_polar(cls, r, phi):
        ''' Returns Point2D object but takes radial distance
            and arg-phi angle which is counter-clockwise
            from (usual X-axis) right part of the horizontal axis
        '''
        x = r * np.cos(np.radians(phi))
        y = r * np.sin(np.radians(phi))
        return cls(x, y)

    # - - - - - - - - - - - - - - - - - - - - - -

    @staticmethod
    def calc_modulus(x, y):
        return np.sqrt(x**2 + y**2)

    @staticmethod
    def calc_argphi(x, y):
        ''' Returns the angle from zero-ray to point(x,y) ray
            counter clockwise from 0 to 360'''
        if x == 0:
            angle = 90 * np.sign(y)
        elif y == 0:
            angle = 0 if x >= 0 else 180
        else:
            angle = np.degrees(np.arctan(y / x))
        if x > 0:
            if y >= 0:
                angle = angle
            else:
                angle = angle + 360
        else:
            angle = angle + 180
        return angle

    # - - - - - - - - - - - - - - - - - - - - - -

    def __str__(self):
        return f"({self.x},{self.y})"

    def __repr__(self):
        return f"Point2D ({self.x},{self.y})"

    def copy(self):
        return self.__class__(self.x, self.y)

    # - - - - - - - - - - - - - - - - - - - - - -

    def __add__(self, otherPoint):
        if isinstance(otherPoint, Point2D):
            add_x = otherPoint.x
            add_y = otherPoint.y
        elif isinstance(otherPoint, tuple | list):
            add_x, add_y = otherPoint
        else:
            raise ValueError("Unknown Point2D coordinates for __add__")
        x_new = self.x + add_x
        y_new = self.y + add_y
        return Point2D(x_new, y_new)

    def __sub__(self, otherPoint):
        if isinstance(otherPoint, Point2D):
            sub_x = otherPoint.x
            sub_y = otherPoint.y
        elif isinstance(otherPoint, tuple | list):
            sub_x, sub_y = otherPoint
        else:
            raise ValueError("Unknown Point2D coordinates for __sub__")
        x_new = self.x - sub_x
        y_new = self.y - sub_y
        return Point2D(x_new, y_new)

    def __mul__(self, value):
        return Point2D(self.x * value, self.y * value)


    @property
    def quarter(self):
        angle = self._PA
        if 0 <= angle <= 180:
            if angle <= 90:
                return 1
            else:
                return 2
        else:
            if angle < 270:
                return 3
            else:
                return 4

    @property
    def radial_distance(self):
        return self._radial_distance

    @property
    def r(self):
        return self._radial_distance

    @property
    def PA(self):
        return self._PA

    @property
    def phi(self):
        return self._PA

    @property
    def argphi(self):
        return self._PA

    @property
    def xy(self):
        return (self.x, self.y)
    
    def rotated(self, PA):
        ''' Returns a Point2D object as (x,y) input point
            but rotated to the angle PA
            around the centre (point with x=0, y=0)
        '''
        PA %= 360
        x = self.x
        y = self.y
        cosPA = np.cos(np.radians(PA))
        sinPA = np.sin(np.radians(PA))
        x_new = x * cosPA - y * sinPA
        y_new = x * sinPA + y * cosPA
        return self.__class__(x_new, y_new)





class LinearFunction2D:
    ''' An instance of the class LinearFunction2D
        allows to use itself as a 2D function y = kx + b
        and can invert this function like x(y) = (y-b)/k
        It handles the case when k is infinity, and it means
        that the function is a straight line with direction
        parallel to Y-axis (vertical axis)
    '''
    def __init__(self, k, b, x_const=None):
        ''' x_const is the value of constant x for
            case when k is infinity and the function
            is a straight line parallel to Y-axis
        '''
        self.k = k
        if not np.isinf(k):
            self.b = b
        else:
            if x_const is None:
                raise ValueError("k is infinity: x_const should be specified")
            self.b = np.nan
        self.x_const = x_const


    def function(self, input_x=None):
        if np.isinf(self.k):
            # does input_x correspond to initial x_const or not (True/False):
            linear_fun = lambda x: x == self.x_const
        elif self.k != 0:
            linear_fun = lambda x: self.k * x + self.b
        else:
            linear_fun = lambda x: self.b            
        if input_x is None:
            return linear_fun
        else:
            return linear_fun(input_x)


    @classmethod
    def from_points(cls, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        dx = x2 - x1
        dy = y2 - y1
        if dx != 0:
            k = dy / dx
        else:
            k = np.inf
        if not np.isinf(k):
            b = y1 - k * x1
        else:
            b = np.nan
        return cls(k, b)


    def __call__(self, input_x=None):
        if input_x is None:
            return self.function
        else:
            return self.function(input_x)


    def __str__(self):
        if np.isinf(self.k):
            return f"x = {self.x_const}"
        elif self.k == 0:
            return f"y = {self.b}"
        else:
            if self.b != 0:
                return f"y = {self.k} * x + {self.b}"
            else:
                return f"y = {self.k} * x"


    def __repr__(self):
        return f"LinearFunction2D {self.__str__()}"

    def copy(self):
        return self.__new__(self.k, self.b)


    def y(self, input_x=None):
        return self.function(input_x)

    def fun(self, input_x=None):
        return self.function(input_x)

    def f(self, input_x=None):
        return self.function(input_x)


    def inverted_function(self, input_y=None):
        ''' invert function of [y(x) = kx + b] is [x(y) = (y - b) / k] '''
        if np.isinf(self.k):
            # this is the case x = x_const always
            linear_fun = lambda y: self.x_const
        elif self.k != 0:
            linear_fun = lambda y: (y - self.b) / self.k
        else:
            # this is the case y = b (const)
            # check if input y == b or not (True/False)
            linear_fun = lambda y: y == self.b
        #
        if input_y is None:
            return linear_fun
        else:
            return linear_fun(input_y)

    def x(self, input_y=None):
        return self.inverted_function(input_y)

    def inverted(self, input_y=None):
        return self.inverted_function(input_y)

    def invert(self, input_y=None):
        return self.inverted_function(input_y)





class LineSegment:
    '''  '''
    def __new__(cls, point1, point2=(0,0)):
        if not isinstance(point1, Point2D | tuple | list):
            raise ValueError("Unknown type of input point1")
        if not isinstance(point2, Point2D | tuple | list):
            raise ValueError("Unknown type of input point2")
        if tuple(point1) == tuple(point2):
            raise ValueError("point 1 is equal to point2 (no line)")
        instance = super(LineSegment, cls).__new__(cls)
        return instance

    def __init__(self, point1, point2=(0,0)):
        x1, y1 = point1
        x2, y2 = point2
        self.point1 = Point2D(x1, y1)
        self.point2 = Point2D(x2, y2)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        #
        linear_fun = LinearFunction2D.from_points(self.point1, self.point2)
        self.linear_function = linear_fun


    def __str__(self):
        return f"{self.point1!s}-{self.point2!s}"


    def __repr__(self):
        return f"LineSegment {self.__str__()}"

    def copy(self):
        return self.__new__(self.point1, self.point2)

    @property
    def x_max(self):
        return max(self.x1, self.x2)

    @property
    def x_min(self):
        return min(self.x1, self.x2)

    @property
    def y_max(self):
        return max(self.y1, self.y2)

    @property
    def y_min(self):
        return min(self.y1, self.y2)


    def on_line(self, point):
        x, y = point
        return self.linear_function(x) == y

    def on_segment(self, point):
        x, y = point
        if not self.x_min <= x <= self.x_max:
            return False
        if not self.y_min <= y <= self.y_max:
            return False
        return self.linear_function(x) == y

    @property
    def function(self):
        return self.linear_function




class Vector2D(LineSegment):
    ''' Vector from point (x_start, y_start) to (x_end, y_end) '''
    #
    def __new__(cls, x_end, y_end, x_start=0, y_start=0):
        return super().__new__(cls, (x_start,y_start), (x_end,y_end))

    def __init__(self, x_end, y_end, x_start=0, y_start=0):
        if int(x_end) != x_end or int(y_end) != y_end:
            x_end = float(x_end)
            y_end = float(y_end)
        else:
            x_end = int(x_end)
            y_end = int(y_end)
        #
        self.x_start = x_start
        self.y_start = y_start
        self._start_point = Point2D(self.x_start, self.y_start)
        self.x_end = x_end
        self.y_end = y_end
        self._end_point = Point2D(self.x_end, self.y_end)
        #
        dx = x_end - x_start
        dy = y_end - y_start
        self.dx = dx
        self.dy = dy
        #
        # PA from 0 to 360 degrees
        self._PA = Point2D.calc_argphi(dx, dy)
        self._modulus = Point2D.calc_modulus(dx, dy)


    @classmethod
    def from_tuple(cls, input_tuple):
        x, y = input_tuple
        return cls(x, y)

    @classmethod
    def radial(cls, point):
        # start_point is (0,0)
        return cls(point.x, point.y)

    @classmethod
    def from_points(cls, point_start, point_end):
        if not isinstance(point_start, Point2D):
            if isinstance(point_start, tuple | list):
                point_start = point_start.from_tuple(point_start)
            else:
                raise ValueError("Unknown type of point_start")
        if not isinstance(point_end, Point2D):
            if isinstance(point_end, tuple | list):
                point_end = point_end.from_tuple(point_end)
            else:
                raise ValueError("Unknown type of point_end")
        x_start, y_start = point_start.xy
        x_end, y_end = point_end.xy
        return cls(x_end, y_end, x_start, y_start)

    # - - - - - - - - - - - - - - - - - - - - - -

    def __str__(self):
        return f"({self.dx},{self.dy})"

    def __repr__(self):
        end_xy = (self.end_point.x, self.end_point.y)
        if self.start_point.x != 0 or self.start_point.y != 0:
            start_xy = (self.start_point.x, self.start_point.y)
            return (f"Vector2D ({self.dx},{self.dy})"
                    f" from {start_xy} to {end_xy}")
        else:
            return f"Vector2D ({self.dx},{self.dy})"

    def __add__(self, otherVector):
        if isinstance(otherVector, Vector2D):
            add_x = otherVector.dx
            add_y = otherVector.dy
        elif isinstance(otherVector, tuple | list):
            add_x, add_y = otherVector
        else:
            raise ValueError("Unknown Vector2D coordinates for __add__")
        x_start = self.x_start
        y_start = self.y_start
        x_end_new = self.x_end + add_x
        y_end_new = self.y_end + add_y
        return Vector2D(x_end_new, y_end_new, x_start, y_start)

    def copy(self):
        return self.__class__.from_points(self.start_point, self.end_point)

    # - - - - - - - - - - - - - - - - - - - - - -

    @property
    def start_point(self):
        return self._start_point

    @start_point.setter
    def start_point(self, *args):
        if len(args) == 1:
            if isinstance(args[0], tuple | list):
                x_start, y_start = args[0]
            elif isinstance(args[0], Point2D):
                x_start, y_start = tuple(args[0])
            else:
                raise TypeError("Unknown type for point in input")
        else:
            x_start, y_start = args
        #
        x_end = self.end_point.x
        y_end = self.end_point.y
        self.__init__(x_end, y_end, x_start, y_start)


    @property
    def end_point(self):
        return self._end_point

    @end_point.setter
    def end_point(self, *args):
        if len(args) == 1:
            if isinstance(args[0], tuple | list):
                x_end, y_end = args[0]
            elif isinstance(args[0], Point2D):
                x_end, y_end = tuple(args[0])
            else:
                raise TypeError("Unknown type for point in input")
        else:
            x_end, y_end = args
        #
        x_start = self.start_point.x
        y_start = self.start_point.y
        self.__init__(x_end, y_end, x_start, y_start)


    @property
    def modulus(self):
        return self._modulus

    @property
    def PA(self):
        return self._PA

    def rotate(self, PA):
        ''' Rotates vector for the angle PA in degrees counter clockwise.
            Rotation around the start_point '''
        dx = self.dx
        dy = self.dy
        cosPA = np.cos(np.radians(PA))
        sinPA = np.sin(np.radians(PA))
        dx_new = dx * cosPA - dy * sinPA
        dy_new = dx * sinPA + dy * cosPA
        x_start = self.x_start
        y_start = self.y_start
        x_end_new = self.x_start + dx_new
        y_end_new = self.y_start + dy_new
        self.__init__(x_end_new, y_end_new, x_start, y_start)

    def rotate_around_00(self, PA):
        x_start = self.x_start
        y_start = self.y_start
        x_end = self.x_end
        y_end = self.y_end
        cosPA = np.cos(np.radians(PA))
        sinPA = np.sin(np.radians(PA))
        x_start_new = x_start * cosPA - y_start * sinPA
        y_start_new = x_start * sinPA + y_start * cosPA
        x_end_new = x_end * cosPA - y_end * sinPA
        y_end_new = x_end * sinPA + y_end * cosPA
        self.__init__(x_end_new, y_end_new, x_start_new, y_start_new)

    def linear_function(self, input_x=None):
        return self.linear_function(input_x) # in superclass


# ------------------------------------------------
# ------------------------------------------------


def extend_array(input_array, Ncells, axis=0):
    ''' Returns a copy of the input_array but with
        zero-filled additional Ncells rows/columns
        If Ncells > 0 then input_array will give
        Ncells new zero-filled rows (or columns)
        If Ncells < 0 then input_array will be extended
        to the left (or down) side.
    '''
    Ncells = int(Ncells)
    if Ncells == 0:
        return input_array
    add_array_shape = list(input_array.shape)
    add_array_shape[axis] = abs(Ncells)
    zeros_array = np.zeros(add_array_shape, dtype=input_array.dtype)
    if Ncells > 0:
        return np.concatenate([input_array, zeros_array], axis=axis)
    else:
        return np.concatenate([zeros_array, input_array], axis=axis)

def crop_array(input_array, Ncells, axis=0):
    ''' Returns a copy of the input_array but with
        Ncells rows (or columns) are removed.
        If Ncells > 0 then input_array will be cropped on Ncells
        from the right (upper) side.
        If Ncells < 0 then input_array will be cropped from
        the left (or down) side.
    '''
    Ncells = int(Ncells)
    if Ncells == 0:
        return input_array
    if Ncells > 0:
        return np.delete(input_array, range(-1,-Ncells-1,-1), axis=axis)
    else:
        return np.delete(input_array, range(abs(Ncells)), axis=axis)



class Shape2D(np.ndarray):
    ''' A numpy 2D-array of type numpy.uint8
        where pixel values are 0 or 1.
        0 corresponds to the emtpy place (out of Shape2D borders)
        1 corresponds to the pixels inside the Shape2D figure
    '''
    _x_cen = None # some figures could have central positions...
    _y_cen = None # not necessary in the array centre — allowed to change
    #
    def __new__(cls, size_y, size_x=None):
        if size_x is None:
            size_x = size_y
        array = super().__new__(cls, (size_y,size_x), dtype=np.uint8)
        array.fill(0)
        return array

    @property
    def size(self):
        return np.max(self.shape)

    @property
    def size_x(self):
        return self.shape[1]

    @property
    def size_y(self):
        return self.shape[0]

    @property
    def array(self):
        return np.array(self)

    @property
    def x_cen(self):
        if self._x_cen is not None:
            return self._x_cen
        else:
            return (self.shape[1] - 1) / 2

    @x_cen.setter
    def x_cen(self, value):
        self._x_cen = value


    @property
    def y_cen(self):
        if self._y_cen is not None:
            return self._y_cen
        else:
            return (self.shape[0] - 1) / 2

    @y_cen.setter
    def y_cen(self, value):
        self._y_cen = value

    @property
    def xy_cen(self):
        return (self.x_cen, self.y_cen)

    @property
    def yx_cen(self):
        return (self.y_cen, self.x_cen)

    def draw_on(self, data, x_cen=None, y_cen=None):
        ''' Input: data array on which the shape (figure)
            should be drawn; x and y central position
            Returns: a copy of the input array but with the figure drawn
        '''
        if x_cen is None:
            x_cen = (data.shape[1] - 1) / 2
        if y_cen is None:
            y_cen = (data.shape[0] - 1) / 2
        #
        shift_x = int(np.round(x_cen - self.x_cen))
        shift_y = int(np.round(y_cen - self.y_cen))
        #
        figure_image = self.copy()
        if shift_x > 0:
            figure_image = extend_array(figure_image, -shift_x, axis=1)
        elif shift_x < 0:
            figure_image = crop_array(figure_image, shift_x, axis=1)
        if shift_y > 0:
            figure_image = extend_array(figure_image, -shift_y, axis=0)
        elif shift_y < 0:
            figure_image = crop_array(figure_image, shift_y, axis=0)
        #
        diff = np.array(data.shape) - np.array(figure_image.shape)
        if diff[1] > 0:
            figure_image = extend_array(figure_image, diff[1], axis=1)
        elif diff[1] < 0:
            figure_image = crop_array(figure_image, -diff[1], axis=1)
        if diff[0] > 0:
            figure_image = extend_array(figure_image, diff[0], axis=0)
        elif diff[0] < 0:
            figure_image = crop_array(figure_image, -diff[0], axis=0)
        #
        new_data = data + figure_image
        return new_data




class Square(Shape2D):
    '''  '''
    #
    def __new__(cls, size):
        array = super().__new__(cls, size, size)
        array.fill(1)
        return array

    @property
    def side_length(self):
        return self.shape[0]

    @property
    def side(self):
        return self.side_length

    @property
    def area(self):
        return self.shape[0]**2




class Rectangle(Shape2D):
    ''' Rectangle shape
        Input: height, width, PA (counter-clockwise)
        where PA is positional angle [-90,90]
    '''
    def __new__(cls, height, width, PA=0.0):
        PA %= 180
        if PA > 90:
            PA = PA - 180
        if PA == 0:
            array = super().__new__(cls, height, width)
            array.fill(1)
            return array
        if PA == 90:
            array = super().__new__(cls, width, height)
            array.fill(1)
            return array
        # ------------------------------------------
        # it is necessary to rotate...
        #
        # lb, lu, ru and rb — left bottom, left upper,
        # right upper and right bottom correspondingly
        # points of the rectangle corners
        lb = Point2D(-width/2, -height/2).rotated(PA)
        lu = Point2D(-width/2, height/2).rotated(PA)
        ru = Point2D(width/2, height/2).rotated(PA)
        rb = Point2D(width/2, -height/2).rotated(PA)
        #
        k = np.tan(np.radians(PA))
        b = lambda x, y, k: y - k * x
        #
        b1 = b(ru.x, ru.y, k)
        upper_line = lambda x: k * x + b1
        b2 = b(rb.x, rb.y, k)
        lower_line = lambda x: k * x + b2
        #
        k2 = -1 / k
        b3 = b(rb.x, rb.y, k2)
        right_line = lambda x: k2 * x + b3
        b4 = b(lb.x, lb.y, k2)
        left_line = lambda x: k2 * x + b4
        #
        if PA > 0:
            leftmost = lu
            topmost = ru
            rightmost = rb
            lowest = lb
            def upper_broken_line(x):
                if x <= topmost.x:
                    return upper_line(x)
                else:
                    return right_line(x)
            def lower_broken_line(x):
                if x <= lowest.x:
                    return left_line(x)
                else:
                    return lower_line(x)
        else:
            leftmost = lb
            topmost = lu
            rightmost = ru
            lowest = rb
            def upper_broken_line(x):
                if x <= topmost.x:
                    return left_line(x)
                else:
                    return upper_line(x)
            def lower_broken_line(x):
                if x <= lowest.x:
                    return lower_line(x)
                else:
                    return right_line(x)
        #   -   -   -   -   -   -   -   -
        array_width = int(np.ceil(rightmost.x - leftmost.x))
        array_height = int(np.ceil(topmost.y - lowest.y))
        array = super().__new__(cls, array_height, array_width)
        x_min = leftmost.x  # negative
        y_min = lowest.y  # negative
        for x in range(array_width):
            corrected_x = x + 0.5 + x_min
            roof = upper_broken_line(corrected_x)
            floor = lower_broken_line(corrected_x)
            yMax = int(np.round(roof + 0.5 - y_min))
            yMin = int(np.round(floor + 0.5 - y_min))
            array[yMin:yMax+1,x] = 1
        return array

    def __init__(self, height, width, PA=0.0):
        self._height = height
        self._width = width
        self._PA = PA

    def __str__(self):
        return f"Rectangle({self._height},{self._width},{self._PA})"

    def __repr__(self):
        output_str = (f"Rectangle (h={self._height}, w={self._width},"
                      f" PA={self._PA})")
        return output_str

    @property
    def height(self):
        return self._height

    @property
    def PA(self):
        return self._PA

    @property
    def wigth(self):
        return self._width

    @property
    def area(self):
        return self.shape[0] * self.shape[1]




class Circle(Shape2D):
    '''  '''
    def __new__(cls, radius):
        diameter = radius * 2
        diameter = int(np.ceil(diameter))
        array = super().__new__(cls, diameter, diameter)
        y_cen, x_cen = (np.array(array.shape) - 1) / 2
        for x in range(diameter):
            y_abs = np.sqrt(radius**2 - (x-x_cen)**2)
            y_min = int(np.round(y_cen-y_abs))
            y_max = int(np.round(y_cen+y_abs))
            array[y_min:y_max+1,x] = 1
        return array

    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @property
    def area(self):
        return np.pi * self.radius**2




class Ellipse(Shape2D):
    ''' Ellipse shape
        Input: ellA — semi-major axis, 
               ellB — semi-minor axis,
               PA — positional angle [-90,90]
    '''
    def __new__(cls, ellA, ellB, PA=0.0):
        # ellA — semi-major-axis along horizontal X-axis
        # ellB — semi-minor-axis along verticzl Y-axis
        if ellA <= 0 or ellB <= 0:
            raise ValueError("ellA and ellB should be positive")
        if ellA == ellB:
            return np.array(Circle(ellA))
        #
        major_axis = ellA * 2
        minor_axis = ellB * 2
        #
        PA %= 180
        if PA > 90:
            PA = PA - 180
        if PA == 0 or PA == 90:
            if PA == 90:
                ellA, ellB = ellB, ellA
                major_axis, minor_axis = minor_axis, major_axis
            array = super().__new__(cls, minor_axis, major_axis)
            y_cen, x_cen = (np.array(array.shape) - 1) / 2
            for x in range(major_axis):
                y_abs = ellB * np.sqrt(1 - ((x-x_cen)/ellA)**2)
                y_min = int(np.round(y_cen-y_abs))
                y_max = int(np.round(y_cen+y_abs))
                array[y_min:y_max+1,x] = 1            
            return array
        # ------------------------------------------
        # it is necessary to rotate...
        #
        # rF0 and lF0 — rigth and left focuses of an ellipse
        # which is not rotated (major axis || X-axis)
        #
        # Consider an ellipse x^2/a^2 + y^2/b^2 = 1
        # with the centre in Point2D(0,0)
        #
        if ellA > ellB:
            a = ellA
            b = ellB
        else:
            # it will be the same shape ellipse but rotate 90 specially
            a = ellB
            b = ellA
            if PA > 0:
                PA -= 90
            else:
                PA += 90
        #
        # to understand the following variables and the sence of the code,
        # search for PDF file with notes about constructing Ellipse shape
        # (attached with the code)
        #
        c = (np.cos(np.radians(PA)))
        s = (np.sin(np.radians(PA)))
        q = b / a
        Q = 1 - q**2
        y_max = a * np.sqrt(1 - Q * c**2)
        x_max = a * np.sqrt(1 - Q * s**2)
        array_height = int(np.ceil(y_max)) * 2
        array_width = int(np.ceil(x_max)) * 2
        array = super().__new__(cls, array_height, array_width)
        #
        C1 = s * c * Q / (1 - Q * s**2)
        C2 = q / (1 - Q * s**2)
        C3 = x_max**2
        #
        def y_1_2(x):
            # Returns two values of y(x)
            x2 = x**2
            if C3 < x2:
                return (np.nan, np.nan)
            term1 = C1 * x
            term2 = C2 * np.sqrt(C3 - x2)
            return (term1 + term2, term1 - term2)
        #
        Xint = int(np.ceil(x_max))
        Yint = int(np.ceil(y_max))
        # fill pixels of the right half of the ellipse
        # Note: pixel centre has coordinate N+0.5
        half_ellipse_1 = [y_1_2(x+0.5) for x in range(0,Xint)]
        # left half of the ellipse is symmetric, need to mirror
        half_ellipse_2 = half_ellipse_1.copy()
        half_ellipse_2.reverse()
        half_ellipse_2 = [(-y2,-y1) for (y1,y2) in half_ellipse_2]
        ellipse_Y = np.array(half_ellipse_2 + half_ellipse_1)
        # shift the ellipse upside 
        ellipse_Y += Yint
        for i in range(array_width):
            roof, floor = ellipse_Y[i]
            if np.isnan(roof):
                # C3 < x**2 in y_1_2 function
                continue
            yMax = int(np.round(roof))
            yMin = int(np.round(floor))
            array[yMin:yMax+1,i] = 1
        return array

    def __init__(self, ellA, ellB, PA=0.0):
        self._ellA = ellA
        self._ellB = ellB
        self._PA = PA




class Polygon(Shape2D):
    ''' Polygon — a shape which is specified by the (x,y) points as corners
        Input: list of pairs (x,y) [Note: x is the first coordinate]
        x and y should be > 0
    '''
    def __new__(cls, input_points_list):
        first_item = input_points_list[0]
        if (isinstance(first_item, tuple | list | np.ndarray)
            and len(first_item) == 2):
            points_list = [Point2D(xy[0],xy[1]) for xy in input_points_list]
        elif isinstance(first_item, Point2D):
            points_list = input_points_list
        else:
            raise TypeError("Unknown type of input points list")
        #
        N = len(points_list)
        x_list = [point.x for point in points_list]
        y_list = [point.y for point in points_list]
        x_max = np.max(x_list)
        x_min = np.min(x_list)
        if x_min < 0:
            raise ValueError("All points should have X coordinate > 0")
        xMax = int(np.ceil(x_max))
        y_max = np.max(y_list)
        y_min = np.min(y_list)
        if y_min < 0:
            raise ValueError("All points should have Y coordinate > 0")
        yMax = int(np.ceil(y_max))
        array = super().__new__(cls, yMax, xMax)
        #
        # ---------------------- ???????????
        #
        segments_list = []
        for i in range(N-1):
            current_point = points_list[i]
            next_point = point_list[i+1]
            segment = LineSegment(current_point, next_point)
            segments_list.append(segment)
        point_first = points_list[0]
        point_last = points_list[-1]
        segment_last = LineSegment(point_last, point_first)
        segments_list.append(segment_last)
        #
        for x in range(xMax+1):
            segments_in_column = dict(
                filter(lambda s: s.x_min < x <= s.x_max,
                       segments_list.items()))
            
        # Create a dictionary of sector parameters of each vector
        # (sector — where the vector is — in polar coordinates ranges)
        sectors_dict = {}
        for i in range(N):
            vector = vectors_list[i]
            start_point = vector.start_point
            end_point = vector.end_point
            phi_range = (start_point.phi, end_point.phi)
            phi_min = min(phi_range)
            phi_max = max(phi_range)
            r_range = (start_point.r, end_point.r) # radial distance
            r_min = min(r_range)
            r_max = max(r_range)
            #
            sector_dict = {}
            sector_dict['phi_min'] = phi_min
            sector_dict['phi_max'] = phi_max
            sector_dict['r_min'] = r_min
            sector_dict['r_max'] = r_max
            #
            sectors_dict[i] = sector_dict
        #
        for x in range(xMax):
            for y in range(yMax):
                point = Point2D(x, y)
                phi = point.phi
                r = point.radial_distance
                #
                filter_phi = lambda item: (item[1]['phi_min'] <= phi 
                                            <= item[1]['phi_max'])
                slice_dict = dict(filter(filter_phi, sectors_dict.items()))
                if len(slice_dict) == 0:
                    continue
                counter = 0
                for i, sector in slice_dict.items():
                    if slice_dict[i]['r_max'] <= r:
                        counter += 1
                    elif slice_dict[i]['r_min'] > r:
                        continue
                    else:
                        vector = vectors_list[i]



                #
                #
                # надо сначала отсеить векторы по phi_min и phi_max,
                # затем у оставшихся нескольких штук посчитать,
                # у скольких векторов r_max меньше point.r;
                # есть ли хоть один вектор, где r_min <= r <= r_max.
                #     если есть, то:
                # вычислить значение r в точке Point2D, которая лежит
                # на радиальном луче к point и на исследуемом векторе
                # — сравнить r с point.r
                #






'''
    Переписать полностью Vector2D и удалить RadialVector2D
    с учетом того, что теперь появился класс LineSegment
    то есть теперь Vector2D будет наследоваться от LineSegment2D
    и тогда Vector2D.__new__ будет обращаться к LineSegment2D.__new__

    Также нужно переписать определение функций lower_line, upper_line,...
    (через создание новых экземпляров) в классе Rectangle
    с учетом того, что появился класс LinearFunction2D
'''

''' 
    Наконец, нужно допилить Polygon.
    Вместо Vector2D можно испольозвать LineSegment,
    хотя и не обязательно, потому что Vector2D всё равно будет наследовать
    Polygon будет строиться так:
    for x in x_range:
        filter: (1) = only segments where x_min <= x <= x_max
        intersections_list = [segment.function(x) for segment in (1)]
        N = len(intersection_list)
        for i in range(0, N, 2):
            yMin = intersections_list[i]
            yMax = intersections_list[i+2] !!! без [N] нужно!!
            array[yMin:yMax+1] = 1
'''










#-==345pt-=32oi5pot2i45-0itvv3\

class MyList(list):
    xlen = 100
    secret = 'tsss'

    def __new__(cls, *args):
        lst = super().__new__(cls, *args)
        return lst 

    def __init__(self, *args):
        self.clear()
        self.extend([*args])
        self.ilen = len(args)
        self.local = 100

    def add_1000(self):
        elems = list(self)
        for i in range(len(elems)):
            elems[i] = elems[i] + 1000
        elems = elems + elems
        self.__init__(*elems)


#[p02oi354tyokerkg0ij3w450-t0i4ewj5y]


class Polynom:
    def __init__(self, *args):
        self.list = args
        self.N = len(args) - 1

    def __repr__(self):
        c_list = reversed(self.list)
        output_str = ''
        for deg, c in enumerate(c_list):
            output_str += f" + {c}x^{self.N-deg}"
        return output_str

    def function(self, x=None):
        result = 0
        for deg, c in enumerate(self.list):
            result += c * x**deg
        return result

    def __call__(self, x=None):
        if x is None:
            return self.function
        else:
            return self.function(x)
