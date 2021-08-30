from src.plants.async_plant import AsyncPlant
from src.plants.controller import Controller


class Runner:
    def __init__(self, plant: AsyncPlant, controller: Controller):
        self.plant = plant
        self.controller = controller

    def run(self):
        self.plant.start()
        while self.plant.is_running():
            state, t = self.plant.get_state()
            u = self.controller.control(state, t)
            self.plant.set_control(u)
