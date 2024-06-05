  from manim import *

class Scene1(Scene):
    def construct(self):
        title = Tex("LaTeX Title").scale(2)
        equation = MathTex("f(x) = x^2 + 1").scale(1.5)
        self.play(Write(title), Write(equation))
        self.play(FadeOut(title), FadeOut(equation))

        grid = NumberPlane()
        grid_title = Tex("Grid").scale(2)
        self.play(Write(grid_title), ShowCreation(grid))
        self.play(FadeOut(grid_title))

        grid.prepare_for_nonlinear_transform()
        grid.apply_function(lambda p: p + 2*p.get_x())
        grid_title = Tex("Transformed Grid").scale(2)
        self.play(Write(grid_title), Transform(grid, grid))
        self.play(FadeOut(grid_title))

class Scene2(Scene):
    def construct(self):
        square = Square()
        self.play(Write(square))
        self.play(Transform(square, Circle()))
        self.play(FadeOut(square))

class Scene3(Scene):
    def construct(self):
        square = Square()
        self.play(Write(square))
        self.play(ApplyPointwiseFunction(lambda p: p + 2*p.get_x(), square))
        self.play(FadeOut(square))

class Scene4(Scene):
    def construct(self):
        text = Text("Text").scale(2)
        equation = MathTex("f(x) = x^2 + 1").scale(1.5)
        self.play(Write(text), Write(equation))
        self.play(FadeOut(text), FadeOut(equation))

class Scene5(Scene):
    def construct(self):
        square = Square()
        decimal = DecimalNumber(0, num_decimal_places=2)
        self.play(Write(square))
        self.play(Write(decimal))
        self.play(decimal.next_to, square, RIGHT)
        self.play(decimal.set_value, square.get_center()[1])
        self.play(FadeOut(square), FadeOut(decimal))

class Scene6(Scene):
    def construct(self):
        shapes = VGroup(
            Circle(),
            Square(),
            Triangle(),
            Rectangle(),
            Polygon(
                np.array([0, 0, 0]),
                np.array([1, 0, 0]),
                np.array([1, 1, 0]),
                np.array([0, 1, 0]),
            ),
        )
        pi_symbol = Tex("\\pi").scale(2)
        self.play(Write(shapes))
        self.play(Write(pi_symbol))
        self.play(SpiralIn(shapes), SpiralIn(pi_symbol))
        self.play(FadeOut(shapes), FadeOut(pi_symbol))

class Scene7(Scene):
    def construct(self):
        triangles = VGroup(
            Triangle(color=RED),
            Triangle(color=GREEN),
            Triangle(color=BLUE),
        )
        self.play(Write(triangles))
        self.play(ApplyMethod(triangles.set_stroke, width=10))
        self.play(ApplyMethod(triangles.set_fill, opacity=0.5))
        self.play(ApplyMethod(triangles.set_stroke, width=0))
        self.play(ApplyMethod(triangles.set_fill, opacity=1))
        self.play(FadeOut(triangles))