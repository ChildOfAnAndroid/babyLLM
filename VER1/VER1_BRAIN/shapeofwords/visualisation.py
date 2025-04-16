from vispy import app, scene
import numpy as np
from VER1_config import *
from cell import Cell
from stats import Stats
from environment import Environment

class Visualisation:
    def __init__(self, stats, environments, signalGrid=None, inertGrid=None, lightGrid=None):
        self.stats = stats
        self.environments = environments
        self.signalGrid = signalGrid
        self.inertGrid = inertGrid
        self.lightGrid = lightGrid

        # Create Vispy canvas and scene
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, title="Game of Why")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=(0, 0, environments.grid.shape[1], environments.grid.shape[0]))
        self.view.camera.interactive = True

        # Add layers for each grid
        self.signal_layer = scene.visuals.Image(self.signalGrid, parent=self.view.scene, cmap='inferno', opacity=0.4)
        print(f"Signal grid unique values: {np.unique(self.signalGrid)}")  # Debug grid values
        self.inert_layer = scene.visuals.Image(self.inertGrid, parent=self.view.scene, cmap='viridis', opacity=0.5)
        print(f"Inert grid unique values: {np.unique(self.inertGrid)}")  # Debug grid values
        self.light_layer = scene.visuals.Image(self.lightGrid, parent=self.view.scene, cmap='plasma')
        print(f"Light grid unique values: {np.unique(self.lightGrid)}")  # Debug grid values

        # Add a layer for cell visualization
        self.cell_layer = scene.visuals.Image(np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32), parent=self.view.scene)
        print(f"Unique cell states: {np.unique([str(cell.state) for row in self.environments.grid for cell in row if isinstance(cell, Cell)])}")

    def update_grid(self):
        """
        Updates the grid data with cell states.
        """
        # Reset cell layer data
        cell_data = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)  # RGBA

        # Process cell data
        for x in range(self.environments.grid.shape[0]):
            for y in range(self.environments.grid.shape[1]):
                cell = self.environments.grid[x, y]
                if isinstance(cell, Cell):
                    color = cell.getCellColor()  # Assuming this returns an RGB tuple
                    alpha = 0.9 if cell.visible else 0  # Adjust transparency based on state
                    cell_data[x, y] = (*color, alpha)

        # Update the cell layer visual
        self.cell_layer.set_data(cell_data)

    def runLoop(self, turn, end=False):
        """
        Updates the visualization for each turn of the simulation.

        Args:
            turn (int): Current turn of the simulation.
            end (bool): Whether this is the final turn.
        """
        if self.signalGrid is not None:
            self.signal_layer.set_data(self.signalGrid)
        if self.inertGrid is not None:
            self.inert_layer.set_data(self.inertGrid)
        if self.lightGrid is not None:
            self.light_layer.set_data(self.lightGrid)

        self.update_grid()
        self.canvas.update()

        # Optionally perform end-of-run tasks
        if end:
            self.endRun(turn)

        # Process events to keep the visualization responsive
        app.process_events()

    def endRun(self, turn):
        """
        Handles tasks needed at the end of the simulation.

        Args:
            turn (int): The final turn of the simulation.
        """
        #print(f"Simulation ended at turn {turn}.")
        # self.canvas.close()