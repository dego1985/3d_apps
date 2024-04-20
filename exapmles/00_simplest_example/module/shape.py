from math import pi, sin, cos

import pyglet
import pyglet.gl as gl
import numpy as np


def create_torus(radius, inner_radius, slices, inner_slices, shader, batch):

    # Create the vertex and normal arrays.
    vertices = []
    normals = []

    u_step = 2 * pi / (slices - 1)
    v_step = 2 * pi / (inner_slices - 1)
    u = 0.0
    for i in range(slices):
        cos_u = cos(u)
        sin_u = sin(u)
        v = 0.0
        for j in range(inner_slices):
            cos_v = cos(v)
            sin_v = sin(v)

            d = radius + inner_radius * cos_v
            x = d * cos_u
            y = d * sin_u
            z = inner_radius * sin_v

            nx = cos_u * cos_v
            ny = sin_u * cos_v
            nz = sin_v

            vertices.extend([x, y, z])
            normals.extend([nx, ny, nz])
            v += v_step
        u += u_step

    # Create a list of triangle indices.
    indices = []
    for i in range(slices - 1):
        for j in range(inner_slices - 1):
            p = i * inner_slices + j
            indices.extend([p, p + inner_slices, p + inner_slices + 1])
            indices.extend([p, p + inner_slices + 1, p + 1])

    # Create a Material and Group for the Model
    diffuse = [0.5, 0.0, 0.3, 1.0]
    ambient = [0.5, 0.0, 0.3, 1.0]
    specular = [1.0, 1.0, 1.0, 1.0]
    emission = [0.0, 0.0, 0.0, 1.0]
    shininess = 50

    material = pyglet.model.Material(
        "custom", diffuse, ambient, specular, emission, shininess
    )
    group = pyglet.model.MaterialGroup(material=material, program=shader)

    vertex_list = shader.vertex_list_indexed(
        len(vertices) // 3,
        gl.GL_TRIANGLES,
        indices,
        batch,
        group,
        position=("f", vertices),
        normals=("f", normals),
        colors=("f", material.diffuse * (len(vertices) // 3)),
    )

    return pyglet.model.Model([vertex_list], [group], batch)


def create_sphere(radius, slices, inner_slices, shader, batch):

    # Create the vertex and normal arrays.
    vertices = []
    normals = []

    u_step = 2 * pi / (slices - 1)
    v_step = 1 * pi / (inner_slices - 1)
    u = 0.0
    for i in range(slices):
        cos_u = cos(u)
        sin_u = sin(u)
        v = 0.0
        for j in range(inner_slices):
            cos_v = cos(v - pi / 2)
            sin_v = sin(v - pi / 2)

            d = radius * cos_v
            x = d * cos_u
            y = d * sin_u
            z = radius * sin_v

            nx = cos_u * cos_v
            ny = sin_u * cos_v
            nz = sin_v

            vertices.extend([x, y, z])
            normals.extend([nx, ny, nz])
            v += v_step
        u += u_step

    # Create a list of triangle indices.
    indices = []
    for i in range(slices - 1):
        for j in range(inner_slices - 1):
            p = i * inner_slices + j
            indices.extend([p, p + inner_slices, p + inner_slices + 1])
            indices.extend([p, p + inner_slices + 1, p + 1])

    # Create a Material and Group for the Model
    diffuse = [0.5, 0.0, 0.3, 1.0]
    ambient = [0.9, 0.9, 0.9, 1.0]
    specular = [1.0, 1.0, 1.0, 1.0]
    emission = [0.0, 0.0, 0.0, 1.0]
    shininess = 50

    material = pyglet.model.Material(
        "custom", diffuse, ambient, specular, emission, shininess
    )
    group = pyglet.model.MaterialGroup(material=material, program=shader)

    vertex_list = shader.vertex_list_indexed(
        len(vertices) // 3,
        gl.GL_TRIANGLES,
        indices,
        batch,
        group,
        position=("f", vertices),
        normals=("f", normals),
        colors=("f", material.diffuse * (len(vertices) // 3)),
    )

    return pyglet.model.Model([vertex_list], [group], batch)


def normalized(xs: np.ndarray):
    return xs / np.linalg.norm(xs, axis=-1, keepdims=True)


def rotation(Ts: np.ndarray, xs: np.ndarray):
    return np.einsum("ijk,imk->imj", Ts[..., :3, :3], xs)


def transform(Ts: np.ndarray, xs: np.ndarray):
    return rotation(Ts, xs) + Ts[:, :3, 3][:, np.newaxis, :]


def create_lines(shader, batch):

    # Create the vertex and normal arrays.

    ts = np.arange(300) / 10
    N_t = len(ts)
    xs = np.stack((np.cos(ts), np.sin(ts), ts * 0.1), axis=1)
    vs = np.stack((-np.sin(ts), np.cos(ts), np.ones(N_t) * 0.1), axis=1)
    us = np.stack((np.cos(ts), np.sin(ts), np.zeros(N_t)), axis=1)

    ezs = normalized(vs)
    exs = normalized(np.cross(ezs, us))
    eys = np.cross(ezs, exs)

    Ts = np.zeros((N_t, 4, 4), np.float32)
    Ts[:, :3, 0] = exs
    Ts[:, :3, 1] = eys
    Ts[:, :3, 2] = ezs
    Ts[:, :3, 3] = xs
    Ts[:, 3, 3] = 1

    r = 0.1
    n_angle = 3
    vertices = []
    normals = []
    for i in range(N_t):
        _vertices = []
        _normals = []
        for angle in range(n_angle + 1):
            x = r * cos(2 * pi / n_angle * angle)
            y = r * sin(2 * pi / n_angle * angle)
            z = 0

            nx = cos(2 * pi / n_angle * angle)
            ny = sin(2 * pi / n_angle * angle)
            nz = 0

            _vertices.append([x, y, z])
            _normals.append([nx, ny, nz])
        vertices.append(_vertices)
        normals.append(_normals)

    vertices = np.array(vertices)
    normals = np.array(normals)

    vertices = transform(Ts, vertices)
    normals = rotation(Ts, normals)

    vertices = vertices.reshape(-1)
    normals = normals.reshape(-1)

    # Create a list of triangle indices.

    indices = []
    for i in range(N_t - 1):
        for angle in range(n_angle):
            p = (n_angle + 1) * i + angle
            indices.extend([p, p + (n_angle + 1) + 1, p + (n_angle + 1)])
            indices.extend([p, p + 1, p + (n_angle + 1) + 1])
    indices = np.array(indices)

    # Create a Material and Group for the Model
    # diffuse = [0.5, 0.0, 0.3, 1.0]
    # ambient = [0.5, 0.0, 0.3, 1.0]
    # specular = [1.0, 1.0, 1.0, 1.0]
    # emission = [0.0, 0.0, 0.0, 1.0]
    # shininess = 50
    ambient = [0.0, 0.0, 0.0, 1.0]
    diffuse = [1.0, 1.0, 1.0, 1.0]
    emission = [0.0, 0.0, 0.0, 1.0]
    specular = [0.0, 0.0, 0.0, 1.0]
    shininess = 10

    material = pyglet.model.Material(
        "custom", diffuse, ambient, specular, emission, shininess
    )
    group = pyglet.model.MaterialGroup(material=material, program=shader)

    vertex_list = shader.vertex_list_indexed(
        len(vertices) // 3,
        gl.GL_TRIANGLES,
        indices,
        batch,
        group,
        position=("f", vertices),
        normals=("f", normals),
        colors=("f", material.diffuse * (len(vertices) // 3)),
    )

    return pyglet.model.Model([vertex_list], [group], batch)
