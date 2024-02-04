#!/usr/bin/env python
import argparse

import pyglet
import pyglet.gl as gl
from pyglet.math import Mat4, Vec3

from module.shape import create_sphere, create_torus


def get_args():
    parser = argparse.ArgumentParser(
        description="This is sample argparse script"
    )
    parser.add_argument(
        "--model_name",
        default="torus",
        type=str,
        help="model name",
    )

    return parser.parse_args()


class App:
    def __init__(self, model_name: str) -> None:
        try:
            # Try and create a window with multisampling (antialiasing)
            config = gl.Config(
                sample_buffers=1, samples=4, depth_size=16, double_buffer=True
            )
            self.window = pyglet.window.Window(
                width=960, height=540, resizable=True, config=config
            )
        except pyglet.window.NoSuchConfigException:
            # Fall back to no multisampling if not supported
            self.window = pyglet.window.Window(
                width=960, height=540, resizable=True
            )
        self.setup()
        self.time = 0.0
        self.batch = pyglet.graphics.Batch()
        shader = pyglet.model.get_default_shader()
        match model_name:
            case "torus":
                self.model = create_torus(1.0, 0.3, 50, 30, shader, self.batch)
            case "sphere":
                self.model = create_sphere(1.0, 30, 15, shader, self.batch)
            case _:
                print(f"{model_name = } is not implemented")
                raise NotImplementedError

        pyglet.clock.schedule_interval(self.update, 1 / 60)

        self.on_draw = self.window.event(self.on_draw)
        self.on_resize = self.window.event(self.on_resize)

    def on_draw(self):
        self.window.clear()
        self.batch.draw()

    def on_resize(self, width, height):
        framebuffer_size = self.window.get_framebuffer_size()

        self.window.viewport = (
            0,
            0,
            framebuffer_size[0] // 2,
            framebuffer_size[1] // 2,
        )
        self.window.projection = Mat4.perspective_projection(
            self.window.aspect_ratio, z_near=0.1, z_far=255, fov=60
        )
        return pyglet.event.EVENT_HANDLED

    def update(self, dt):
        self.time += dt
        rot_x = Mat4.from_rotation(self.time, Vec3(1, 0, 0))
        rot_y = Mat4.from_rotation(self.time / 2, Vec3(0, 1, 0))
        rot_z = Mat4.from_rotation(self.time / 4, Vec3(0, 0, 1))
        trans = Mat4.from_translation(Vec3(0.0, 0.0, -3.0))
        self.model.matrix = trans @ rot_x @ rot_y @ rot_z

    def setup(self):
        # One-time GL setup
        gl.glClearColor(1, 1, 1, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        self.on_resize(*self.window.size)

        # Uncomment this line for a wireframe view:
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)


if __name__ == "__main__":
    args = get_args()
    print(f"{args = }")
    App(**vars(args))
    pyglet.app.run()
