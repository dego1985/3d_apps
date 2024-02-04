from math import pi, sin, cos

import pyglet
import pyglet.gl as gl


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
