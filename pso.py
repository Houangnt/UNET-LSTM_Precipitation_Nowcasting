import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.zeros(dim)
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

class PSO:
    def __init__(self, dim, bounds, num_particles, max_iter, w=0.5, c1=2, c2=2):
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.swarm = [Particle(dim, bounds) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(bounds[0], bounds[1], dim)
        self.global_best_score = float('inf')

    def optimize(self, obj_func):
        for i in range(self.max_iter):
            for particle in self.swarm:
                fitness = obj_func(particle.position)
                if fitness < particle.best_score:
                    particle.best_score = fitness
                    particle.best_position = np.copy(particle.position)
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = np.copy(particle.position)

            for particle in self.swarm:
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                particle.velocity = (self.w * particle.velocity) + \
                                    (self.c1 * r1 * (particle.best_position - particle.position)) + \
                                    (self.c2 * r2 * (self.global_best_position - particle.position))
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

        return self.global_best_position, self.global_best_score
