# MAIN FILE: GAME OF WHY
# CHARIS CAT 2024

import cProfile
import pstats

from vispy.app import Timer, run as vispy_run

#from automaton import *
#from environment import *
#from stats import *
from visualisation import *

class Main:
    """

    pre game:
    initialise environment
    initialise cells


    each turn:
    revisualise
    move cells
    refresh environment

    end of turn:
    print results

    end of game:
    print results

    """

    def __init__(self):
        self.stats = Stats()
        self.environments = Environment(self.stats)
        self.automaton = Automaton(self.stats, self.environments)
        self.visualisation = Visualisation(self.stats, self.environments)
        self.simulationRecorder = SimulationRecorder()

        # Timer setup for simulation loop
        self.turn = 0
        self.timer = Timer(0.1, connect=self.run, iterations=-1, start=True)
        self.fastForward = False

    def run(self, event=None):
        if self.turn > NUM_STEPS:
            if self.timer.running:
                self.timer.stop()
            self.stats.endRun()
            self.visualisation.endRun(NUM_STEPS)
            self.simulationRecorder.end()
            return
        
        #try:
        if self.fastForward:
            while self.turn <= NUM_STEPS:
                self.timer.stop()
                self.runLoop(self.turn, self.turn == NUM_STEPS)
        else:
            if not self.timer.running:
                self.timer.start()
            self.runLoop(self.turn, self.turn == NUM_STEPS)
        #except Exception as e:
        #    print(f"Exception in runLoop: {e}")
        #    if self.timer.running:
        #        self.timer.stop()

    def runLoop(self, turn, end):
        print(f"turn {self.turn} starting!")
        print(f"Timer running: {self.timer.running}")
        self.stats.beginTurn()
        self.environments.runLoop(turn)
        self.visualisation.runLoop(turn, end=end)
        self.automaton.runLoop(turn)
        self.stats.endTurn()
        self.simulationRecorder.endTurn()
        self.turn += 1
        print(f"turn {self.turn} over!")
        print(f"Timer running: {self.timer.running}")

# Profiling Block
if __name__ == "__main__":
    # Initialize the profiler
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling

    # Run your simulation
    main = Main()
    main.run()
    vispy_run()

    # Stop profiling
    profiler.disable()

    # Print profiling stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime').print_stats(10)  # Top 10 time-consuming functions