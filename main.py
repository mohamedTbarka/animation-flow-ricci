import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import cv2
import time


class RicciFlowPlus:
    def __init__(self):
        pygame.init()
        self.display = (1024, 768)
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Team Plus")

        # Initialize Pygame font with smaller size
        pygame.font.init()
        self.font = pygame.font.Font(None, 48)  # Reduced size

        # Create text surface with adjusted color
        self.text_surface = self.font.render("TEAM PLUS", True, (255, 255, 255))
        self.text_data = pygame.image.tostring(self.text_surface, "RGBA", True)
        self.text_width = self.text_surface.get_width()
        self.text_height = self.text_surface.get_height()

        # Generate texture for text
        self.text_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.text_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.text_width, self.text_height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, self.text_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Set up 3D viewing with adjusted camera
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -10)

        # Enable OpenGL features
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Set light properties
        glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 5, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))

        # Add shape preservation parameters
        self.original_vertices = None
        self.max_deformation = 0.5
        self.shape_memory = 0.3

        # Animation parameters
        self.time = 0
        self.rotation = {'x': 0, 'y': 0, 'z': 0}

        # Video recording parameters
        self.record_video = True
        self.video_duration = 60  # 60 seconds
        self.fps = 60
        self.frame_count = 0
        self.total_frames = self.video_duration * self.fps

        # Initialize video writer
        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                'team_plus_animation.mp4',
                fourcc,
                self.fps,
                (self.display[0], self.display[1])
            )

        # Initialize meshes
        self.initialize_mesh()
        self.create_outline_plus()

    def initialize_mesh(self):
        """Initialize mesh with shape preservation"""
        self.resolution = 15

        # Create vertices for horizontal bar
        h_vertices = []
        for i in range(self.resolution):
            for j in range(self.resolution):
                x = -1.5 + 3.0 * i / (self.resolution - 1)
                y = -0.3 + 0.6 * j / (self.resolution - 1)
                z = 0.2 + np.random.normal(0, 0.01)
                h_vertices.append([x, y, z])

        # Create vertices for vertical bar
        v_vertices = []
        for i in range(self.resolution):
            for j in range(self.resolution):
                x = -0.3 + 0.6 * i / (self.resolution - 1)
                y = -1.5 + 3.0 * j / (self.resolution - 1)
                z = 0.2 + np.random.normal(0, 0.01)
                v_vertices.append([x, y, z])

        # Combine vertices
        self.vertices = np.array(h_vertices + v_vertices, dtype=np.float32)
        self.original_vertices = self.vertices.copy()
        self.curvatures = np.zeros(len(self.vertices))

        # Generate faces
        self.faces = []

        # Faces for horizontal bar
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                idx = i * self.resolution + j
                self.faces.extend([
                    [idx, idx + 1, idx + self.resolution],
                    [idx + 1, idx + self.resolution + 1, idx + self.resolution]
                ])

        # Faces for vertical bar
        base_idx = self.resolution * self.resolution
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                idx = base_idx + i * self.resolution + j
                self.faces.extend([
                    [idx, idx + 1, idx + self.resolution],
                    [idx + 1, idx + self.resolution + 1, idx + self.resolution]
                ])

        self.faces = np.array(self.faces)
        self.neighbor_indices = self.precompute_neighbors()

    def create_outline_plus(self):
        """Create vertices for the stable plus outline"""
        thickness = 0.4
        size = 2.0
        depth = 0.1

        self.outline_vertices = np.array([
            # Horizontal bar
            [-size, -thickness, depth], [size, -thickness, depth],
            [size, thickness, depth], [-size, thickness, depth],
            [-size, -thickness, -depth], [size, -thickness, -depth],
            [size, thickness, -depth], [-size, thickness, -depth],

            # Vertical bar
            [-thickness, -size, depth], [thickness, -size, depth],
            [thickness, size, depth], [-thickness, size, depth],
            [-thickness, -size, -depth], [thickness, -size, -depth],
            [thickness, size, -depth], [-thickness, size, -depth]
        ], dtype=np.float32)

    def precompute_neighbors(self):
        """Compute neighboring vertices for each vertex"""
        neighbors = [[] for _ in range(len(self.vertices))]
        for face in self.faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                neighbors[v1].append(v2)
                neighbors[v2].append(v1)
        return [list(set(n)) for n in neighbors]

    def compute_ricci_flow(self):
        """Compute constrained Ricci flow to preserve shape"""
        dt = 0.01

        # Compute curvatures
        for i in range(len(self.vertices)):
            neighbors = self.neighbor_indices[i]
            if len(neighbors) < 2:
                continue

            angle_sum = 0
            for j in range(len(neighbors)):
                v1 = self.vertices[neighbors[j]] - self.vertices[i]
                v2 = self.vertices[neighbors[(j + 1) % len(neighbors)]] - self.vertices[i]

                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)

                if v1_norm < 1e-7 or v2_norm < 1e-7:
                    continue

                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
                angle_sum += np.arccos(cos_angle)

            self.curvatures[i] = np.clip(2 * np.pi - angle_sum, -2 * np.pi, 2 * np.pi)

        # Update vertices with shape preservation
        for i in range(len(self.vertices)):
            if len(self.neighbor_indices[i]) < 2:
                continue

            # Compute normal
            normal = np.zeros(3)
            for j in range(len(self.neighbor_indices[i])):
                v1 = self.vertices[self.neighbor_indices[i][j]] - self.vertices[i]
                v2 = self.vertices[self.neighbor_indices[i][(j + 1) % len(self.neighbor_indices[i])]] - self.vertices[i]
                normal += np.cross(v1, v2)

            norm = np.linalg.norm(normal)
            if norm > 1e-7:
                normal /= norm

                # Ricci flow movement
                flow_movement = -self.curvatures[i] * normal * dt

                # Shape preservation force
                to_original = self.original_vertices[i] - self.vertices[i]
                shape_force = self.shape_memory * to_original

                # Combine movements
                total_movement = flow_movement + shape_force

                # Limit maximum movement
                movement_magnitude = np.linalg.norm(total_movement)
                if movement_magnitude > self.max_deformation:
                    total_movement = total_movement * self.max_deformation / movement_magnitude

                self.vertices[i] += total_movement

    def draw_mesh(self):
        """Draw the evolving mesh with enhanced visibility"""
        glPushMatrix()

        glRotatef(self.rotation['x'], 1, 0, 0)
        glRotatef(self.rotation['y'], 0, 1, 0)

        # Draw mesh
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            for vertex_idx in face:
                color_val = (self.curvatures[vertex_idx] + 2 * np.pi) / (4 * np.pi)
                glColor4f(1.0, 0.4 + 0.6 * color_val, 0.2, 0.8)
                glVertex3fv(self.vertices[vertex_idx])
        glEnd()

        # Draw wireframe
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(1.0)
        glColor4f(1.0, 1.0, 1.0, 0.2)

        glBegin(GL_TRIANGLES)
        for face in self.faces:
            for vertex_idx in face:
                glVertex3fv(self.vertices[vertex_idx])
        glEnd()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glPopMatrix()

    def draw_outline_plus(self):
        """Draw enhanced outline plus"""
        glPushMatrix()

        glRotatef(self.rotation['x'], 1, 0, 0)
        glRotatef(self.rotation['y'], 0, 1, 0)

        # Draw filled plus
        glColor4f(1.0, 0.2, 0.2, 0.4)

        glBegin(GL_QUADS)
        for i in range(0, 8, 4):
            for j in range(4):
                glVertex3fv(self.outline_vertices[i + j])

        for i in range(8, 16, 4):
            for j in range(4):
                glVertex3fv(self.outline_vertices[i + j])
        glEnd()

        # Draw outline
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(2.0)
        glColor4f(1.0, 1.0, 1.0, 0.8)

        glBegin(GL_QUADS)
        for i in range(0, 16, 4):
            for j in range(4):
                glVertex3fv(self.outline_vertices[i + j])
        glEnd()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glPopMatrix()

    def render_text(self):
        """Render text using pygame texture with adjusted position and size"""
        glPushMatrix()

        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.text_texture)

        # Adjusted position and size
        glTranslatef(-1.0, 2.0, 0)
        aspect_ratio = self.text_width / self.text_height
        scale = 2.0

        glBegin(GL_QUADS)
        glColor4f(1, 1, 1, 1)
        glTexCoord2f(0, 0);
        glVertex3f(0, 0, 0)
        glTexCoord2f(1, 0);
        glVertex3f(scale * aspect_ratio, 0, 0)
        glTexCoord2f(1, 1);
        glVertex3f(scale * aspect_ratio, scale * 0.3, 0)
        glTexCoord2f(0, 1);
        glVertex3f(0, scale * 0.3, 0)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)

        glPopMatrix()

    def capture_frame(self):
        """Capture the current frame for video recording"""
        # Read the pixels from the OpenGL buffer
        string_image = pygame.image.tostring(self.screen, 'RGB')
        temp_surf = pygame.image.fromstring(string_image, self.display, 'RGB')

        # Convert Pygame surface to numpy array for OpenCV
        image_array = pygame.surfarray.array3d(temp_surf)
        image_array = image_array.swapaxes(0, 1)

        # Convert from RGB to BGR for OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        return image_array

    def run(self):
        clock = pygame.time.Clock()
        start_time = time.time()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if self.record_video:
                        self.video_writer.release()
                    pygame.quit()
                    return

            self.rotation['x'] += 1.0
            self.rotation['y'] += 1.5

            self.compute_ricci_flow()

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0.05, 0.05, 0.1, 1)

            self.draw_mesh()
            self.draw_outline_plus()
            self.render_text()

            pygame.display.flip()

            # Record frame if video recording is enabled
            if self.record_video:
                frame = self.capture_frame()
                self.video_writer.write(frame)
                self.frame_count += 1

                # Print progress
                if self.frame_count % 60 == 0:  # Show progress every second
                    print(f"Recording progress: {self.frame_count / self.total_frames * 100:.1f}%")

                # Check if we've recorded enough frames
                if self.frame_count >= self.total_frames:
                    print("Recording completed!")
                    self.video_writer.release()
                    pygame.quit()
                    return

            clock.tick(self.fps)

if __name__ == "__main__":
    print("Starting Team Plus Animation Recording...")
    print("Recording a 60-second video at 60 FPS...")
    app = RicciFlowPlus()
    app.run()