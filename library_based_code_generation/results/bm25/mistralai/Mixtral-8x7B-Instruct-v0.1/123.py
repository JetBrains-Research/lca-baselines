 from manim import *

class Scene1(Scene):
def construct(self):
title = Tex("Scene 1", font_size=60).to_edge(UP)
equation = Tex(r"f(x) = x^2", font_size=40).to_edge(DOWN)

self.play(Write(title))
self.play(Write(equation))
self.wait(1)

self.play(Transform(title, Tex("Transformed Title")),
FadeOut(equation))

grid = NumberPlane()
grid_title = Tex("Grid", font_size=60).to_edge(UP)

self.play(Create(grid), Write(grid_title))
self.wait(1)

grid_func = lambda p: p.rotate(PI / 4) * np.array([1, 1, 0])
grid_transformed = grid.apply_function(grid_func)

self.play(Transform(grid, grid_transformed),
Transform(grid_title, Tex("Transformed Grid")))
self.wait(1)

class Scene2(Scene):
def construct(self):
square = Square()
circle = Circle()

self.play(Create(square))
self.wait(1)

self.play(Transform(square, circle))
self.wait(1)

self.play(FadeOut(circle))

class Scene3(Scene):
def construct(self):
square = Square()
pointwise_func = lambda p: p + np.array([0.2, 0.3, 0])

square_transformed = square.apply_function(pointwise_func)

self.play(Create(square))
self.wait(1)

self.play(Transform(square, square_transformed))
self.wait(1)

class Scene4(Scene):
def construct(self):
title = Tex("Scene 4", font_size=60).to_edge(UP)
equation = Tex(r"g(x) = \sin(x)", font_size=40).to_edge(DOWN)

self.play(Write(title))
self.play(Write(equation))
self.wait(1)

class Scene5(Scene):
def construct(self):
square = Square()
decimal = DecimalNumber(0, font_size=40).next_to(square, RIGHT)

self.play(Create(square), Write(decimal))
self.wait(1)

self.play(Test(decimal, WatchTheDecimals()))
self.wait(1)

class Scene6(Scene):
def construct(self):
shapes = VGroup(*[Circle(), Square(), Star()])
pi_symbol = Tex("π", font_size=120).move_to(ORIGIN)

self.play(Create(shapes), Write(pi_symbol))
self.wait(1)

self.play(Test(shapes, CubicAndLineto(0.5)), FadeOut(pi_symbol))
self.wait(1)

class Scene7(Scene):
def construct(self):
triangle1 = Polygon([0, 0, 0], [1, 0, 0], [0, 1, 0])
triangle2 = Polygon([0, 0, 0], [1, 0, 0], [0, -1, 0])
triangle3 = Polygon([0, 0, 0], [-1, 0, 0], [0, 1, 0])

self.play(Create(triangle1), Create(triangle2), Create(triangle3))
self.wait(1)

self.play(triangle1.animate.set_line_to_style(dash_ratio=0.2),
triangle2.animate.set_line_to_style(dash_ratio=0.5),
triangle3.animate.set_line_to_style(dash_ratio=0.8))
self.wait(1)

self.play(FadeOut(triangle1, triangle2, triangle3))