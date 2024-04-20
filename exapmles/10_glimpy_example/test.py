#!/usr/bin/env python
import numpy as np
from glumpy import app, gl, glm, gloo
import modules.dual_quaternion as dq

vertex = """
uniform vec4 ucolor;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
attribute vec3 position;
attribute vec4 color;

varying vec4 v_color;

void main()
{
    v_color = ucolor * color;
    gl_Position = projection * view * model * vec4(position,1.0);
}
"""

fragment = """
varying vec4 v_color;
void main()
{
    gl_FragColor = v_color;
}
"""


class Arrows:
    def __init__(self, pose_buffer_size: int) -> None:
        self.pose_buffer_size = pose_buffer_size
        self.pose_index = 0

        vtype = [
            ("position", np.float32, 3),
            ("color", np.float32, 4),
        ]
        itype = np.uint32

        # 矢印形状を定義
        self.arrow = 0.1 * np.array(
            [[0, 0, 1], [-1, 0, 0], [1, 0, 0]], np.float32
        )

        # 矢印の表と裏の色を定義
        self.colors = np.array([[1, 0, 0, 1], [0, 0, 1, 1]])

        # 矢印の面の頂点の相対インデックス
        self.faces_p = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]

        # 矢印の面の頂点の色のインデックス
        self.faces_c = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

        # 頂点の数
        self.max_vertex_index = len(self.faces_p) * self.pose_buffer_size
        vertices = np.zeros(self.max_vertex_index, vtype)

        # 面の頂点のインデックスの配列
        # 初期値として、どの頂点も示さないインデックスを設定
        faces = np.full(
            24 * pose_buffer_size, self.max_vertex_index, dtype=itype
        )
        # アウトラインの頂点のインデックスの配列
        outline = np.full(
            12 * pose_buffer_size, self.max_vertex_index, dtype=itype
        )

        self.vertices = vertices.view(gloo.VertexBuffer)
        self.faces = faces.view(gloo.IndexBuffer)
        self.outline = outline.view(gloo.IndexBuffer)

        self.obj = gloo.Program(vertex, fragment)
        self.obj.bind(self.vertices)
        self.obj["model"] = np.eye(4, dtype=np.float32)
        self.obj["view"] = glm.translation(0, 0, -5)

    def push(self, new_pose_dq: dq.ndarray, pose_dq: dq.ndarray):
        self._set_index_status(self.pose_index, True)
        self._set_pose(new_pose_dq, pose_dq, self.pose_index)
        self._uodate_index()

    def draw(self):
        # Filled obj
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glEnable(gl.GL_CULL_FACE)
        self.obj["ucolor"] = 1, 1, 1, 1
        self.obj.draw(gl.GL_TRIANGLES, self.faces)

        # Outlined obj
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glEnable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_FALSE)
        self.obj["ucolor"] = 0, 0, 0, 1
        self.obj.draw(gl.GL_LINES, self.outline)
        gl.glDepthMask(gl.GL_TRUE)

        # Make obj rotate
        model = np.eye(4, dtype=np.float32)
        # glm.rotate(model, theta, 0, 0, 1)
        # glm.rotate(model, -phi, 1, 0, 0)
        self.obj["model"] = model

    def update_projection(self, width, height):
        self.obj["projection"] = glm.perspective(
            45.0, width / float(height), 2.0, 100.0
        )

    def _uodate_index(self):
        self.pose_index += 1
        self.pose_index %= self.pose_buffer_size

    def _get_faces_index(self, pose_index):
        target_faces = self.faces[24 * pose_index : (pose_index + 1) * 24]
        return target_faces

    def _get_outline_index(self, pose_index):
        target_outline = self.outline[pose_index * 12 : (pose_index + 1) * 12]
        return target_outline

    def _set_pose(
        self, new_pose_dq: dq.ndarray, pose_dq: dq.ndarray, pose_index: int
    ):
        # verticesを更新
        arrow0 = pose_dq.transform(self.arrow)
        arrow1 = new_pose_dq.transform(self.arrow)
        offset = pose_index * 12
        self.vertices["position"][offset : offset + 3] = arrow0
        self.vertices["position"][offset + 6 : offset + 9] = arrow0
        self.vertices["position"][offset + 3 : offset + 6] = arrow1
        self.vertices["position"][offset + 9 : offset + 12] = arrow1
        self.vertices["color"][offset : offset + 6] = [1, 0, 0, 1]
        self.vertices["color"][offset + 6 : offset + 12] = [0, 0, 1, 1]

    def _set_index_status(self, pose_index: int, visible: bool):
        itype = np.uint32
        vertex_index_offset = 12 * pose_index
        target_faces = self._get_faces_index(pose_index)
        target_outline = self._get_outline_index(pose_index).view()
        if visible:
            target_faces[:] = (
                np.array(
                    [
                        [0, 3, 4, 0, 4, 1, 0, 2, 5, 0, 5, 3],
                        np.array([0, 3, 4, 0, 4, 1, 0, 2, 5, 0, 5, 3])[::-1]
                        + 6,
                    ],
                    dtype=itype,
                ).flatten()
                + vertex_index_offset
            ) % self.max_vertex_index
            target_outline[:] = (
                np.array([0, 2, 2, 5, 5, 3, 3, 4, 4, 1, 1, 0], dtype=itype)
                + vertex_index_offset
            ) % self.max_vertex_index
        else:
            target_faces[:] = np.full(24, self.max_vertex_index, dtype=itype)
            target_outline[:] = np.full(12, self.max_vertex_index, dtype=itype)


class Motion:
    def __init__(self, pose: dq.ndarray) -> None:
        self.pose = pose

    def update(self):
        pitch = dq.rotx(0.1)
        roll = dq.rotz(0.1 * np.random.randn())
        vdt = dq.transz(0.04)

        self.pose = self.pose * vdt * roll * pitch


class Window:
    def __init__(self) -> None:
        self.window = app.Window(
            width=1024, height=1024, color=(0.30, 0.30, 0.35, 1.00)
        )

        self.motions = []
        for _ in range(10):
            pose = dq.rotx(2 * np.pi * np.random.rand(1)[0])
            self.motions.append(Motion(pose))

        self.arrows = Arrows(100)

        self.window.event(self.on_draw)
        self.window.event(self.on_resize)
        self.window.event(self.on_init)

    def run(self):
        app.run()

    def on_draw(self, dt):
        self.window.clear()

        for motion in self.motions:
            pose0 = motion.pose.copy()
            motion.update()
            pose1 = motion.pose.copy()
            self.arrows.push(pose1, pose0)

        self.arrows.draw()

    def on_resize(self, width, height):
        self.arrows.update_projection(width, height)

    def on_init(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glPolygonOffset(1, 1)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glLineWidth(4.0)


win = Window()
win.run()
