# CELL CLASS FILE: GAME OF WHY
# CHARIS CAT 2024

from matplotlib.colors import hsv_to_rgb
import random
from VER1_config import *
from simulation_recorder import SimulationRecorder

class Cell:
    topEnergy = 1
    CellAttractivenessTopRecord = CELL_ATTRACTIVENESS_TOP_RECORD_INIT
    ratioResult = 0
    attractivenessGain = 0
    def __init__(self, x, y, stats, environment, organismCheck=None, parent=None):
        self.generalStatsList = ["growthRate",
        "resilience",
        "perception",
        "speed",
        "lightEmission",
        "lightAbsorption",
        "inertEmission",
        "inertAbsorption",
        "lifeExpectancyMin",
        "lifeExpectancyMax",
        "fertilityRate",
        "fertilityAgeMin",
        "fertilityAgeMax",
        "fertilityEnergy",
        "mass",
        "height",
        "lightStorage",
        "energyStorage",
        "inertStorage",
        "mutationRate"]
        self.id = stats.getCellNextID() # Cell ID
        self.alive = True
        self.visible = True
        self.environment = environment
        self.age = 0  # cell age (in turns)
        self.role = "general"  # Role of the cell: general, structural, sensory, reproductive
        self.organism = organismCheck  # Tracks which organism this cell belongs to
        self.parent = parent
        self.stats = stats
        self.attractiveness = CELL_BASE_ATTRACTIVENESS_MIN
        self.growthDecayRate = CELL_BASE_GROWTH_DECAY_RATE
        self.x = x # position x
        self.y = y # position y
        self.memory = []
        self.previousAlive = 0
        self.topEnergyDecay = 0
        self.luck = 0 # UNIMPLEMENTED: range -100 to 100
        self.cellEnergyRecord = 0
        self.moveLoopCounter = 0
        self.prevX = x
        self.prevY = y
        self.turnRoll = 1
        self.turnRollAlt = 1
        self.tightTurnRoll = 1
        # random.choices([-1, 1], weights = [(self.luck + 100)/200), (1-((self.luck + 100)/200)]) # Luck (assuming scaled -100 to 100) and a random chance weight the + or - choice

        if parent is None:
            self.spawnNew()
        else:
            self.spawnChild(parent)
        self.saveBirthStats()
        SimulationRecorder().recordBirth(self)

    def getTurnInfo(self):
        self.turnRoll = random.uniform(0.7, 1.3)
        self.turnRollAlt = random.uniform(0.8, 1.2)
        self.tightTurnRoll = random.uniform(0.95, 1.05)
        self.inertUnderCell = self.environment.getInertAt(self.x, self.y)

    def saveBirthStats(self):
        onBirthStats = (f"\n Hey, Cell {self.id} here. Just passing on my birth certificate! Born to {self.parent} on turn {self.turnCount}, at {self.x},{self.y}. Cell role: {self.role}. Attractiveness: {self.attractiveness}. Growth Decay Rate: {self.growthDecayRate}. Luck: {self.luck}. Highest Energy: {self.cellEnergyRecord}. Energy: {self.energy}. Growth Rate: {self.growthRate}. Resilience: {self.resilience}. Perception Strength: {self.perception}. Speed: {self.speed}. Light Emission: {self.lightEmission}. Light Absorption: {self.lightAbsorption}. Mutation Rate: {self.mutationRate}. Life Expectancy: {self.lifeExpectancy}. Fertility Rate: {self.fertilityRate}. Fertility Age: {self.fertilityAgeMin} - {self.fertilityAgeMax}. Energy needed for reproduction: {self.fertilityEnergy}. Mass: {self.mass}. Height: {self.height}. Colour: {self.color}.")
            
        with open("birthDeathStats.txt", "a") as file:
            file.write(onBirthStats + "\n")

        #print(f"Cell {self.id} birth written to birthDeathStats.txt successfully!")
        
    def spawnNew(self):
        self.turnCount = 0
        self.state = random.choice(list(CellState))
        print (f"{self.state} its me") # Starting state

        # INDIVIDUAL CELL BEHAVIOURS
        match self.state:
            case CellState.PLASMA:
                self.growthRate = random.uniform(CELL_PLASMA_GROWTHRATE_MIN, CELL_PLASMA_GROWTHRATE_MAX)  # Energy Absorption
                self.resilience = random.uniform(CELL_PLASMA_RESILIENCE_MIN, CELL_PLASMA_RESILIENCE_MAX)  # 'toughness'
                self.perception = random.uniform(CELL_PLASMA_PERCEPTION_MIN, CELL_PLASMA_PERCEPTION_MAX)  # Sensory acuity
                self.speed = random.uniform(CELL_PLASMA_SPEED_MIN, CELL_PLASMA_SPEED_MAX)  # movement speed
                self.lightEmission = random.uniform(CELL_PLASMA_LIGHTEMISSION_MIN, CELL_PLASMA_LIGHTEMISSION_MAX)  # Amount of light emitted (e.g., by plasma or bioluminescence)
                self.lightAbsorption = random.uniform(CELL_PLASMA_LIGHTABSORPTION_MIN, CELL_PLASMA_LIGHTABSORPTION_MAX)  # Ability to absorb light as energy
                self.inertEmission = random.uniform(CELL_PLASMA_INERTEMISSION_MIN, CELL_PLASMA_INERTEMISSION_MAX)
                self.inertAbsorption = random.uniform(CELL_PLASMA_INERTABSORPTION_MIN, CELL_PLASMA_INERTABSORPTION_MAX)
                self.mutationRate = random.uniform(CELL_PLASMA_MUTATIONRATE_MIN, CELL_PLASMA_MUTATIONRATE_MAX)  # Probability of mutation during reproduction
                self.lifeExpectancyMin = random.uniform(CELL_PLASMA_LIFEEXPECTANCYMIN_MIN, CELL_PLASMA_LIFEEXPECTANCYMIN_MAX)
                self.lifeExpectancyMax = random.uniform(CELL_PLASMA_LIFEEXPECTANCYMAX_MIN, CELL_PLASMA_LIFEEXPECTANCYMAX_MAX)
                self.fertilityRate = random.uniform(CELL_PLASMA_FERTILITYRATE_MIN, CELL_PLASMA_FERTILITYRATE_MAX)
                self.fertilityAgeMin = random.uniform(CELL_PLASMA_FERTILITYAGEMIN_MIN, CELL_PLASMA_FERTILITYAGEMIN_MAX)
                self.fertilityAgeMax = random.uniform(CELL_PLASMA_FERTILITYAGEMAX_MIN, CELL_PLASMA_FERTILITYAGEMAX_MAX)
                self.fertilityEnergy = random.uniform(CELL_PLASMA_FERTILITYENERGY_MIN, CELL_PLASMA_FERTILITYENERGY_MAX)
                self.mass = random.uniform(CELL_PLASMA_MASS_MIN, CELL_PLASMA_MASS_MAX)
                self.height = random.uniform(CELL_PLASMA_HEIGHT_MIN, CELL_PLASMA_HEIGHT_MAX)
                self.prefHeight = random.uniform(CELL_PLASMA_PREFHEIGHT_MIN, CELL_PLASMA_PREFHEIGHT_MAX)
                self.lightStorage = random.uniform (CELL_PLASMA_LIGHTSTORAGE_MIN, CELL_PLASMA_LIGHTSTORAGE_MAX)
                self.energyStorage = random.uniform(CELL_PLASMA_ENERGYSTORAGE_MIN, CELL_PLASMA_ENERGYSTORAGE_MAX)
                self.inertStorage = random.uniform(CELL_PLASMA_INERTSTORAGE_MIN, CELL_PLASMA_INERTSTORAGE_MAX)
                self.birthColor = random.uniform(CELL_PLASMA_COLOR_MIN, CELL_PLASMA_COLOR_MIN)
                self.color = self.birthColor
                self.setEnergy(random.uniform(CELL_PLASMA_ENERGY_MIN, CELL_PLASMA_ENERGY_MAX), "spawnNew PLASMA")
                # self.energy = random.uniform(CELL_PLASMA_ENERGY, CELL_PLASMA_ENERGY)
            case CellState.GAS:
                self.growthRate = random.uniform(CELL_GAS_GROWTHRATE_MIN, CELL_GAS_GROWTHRATE_MAX)
                self.resilience = random.uniform(CELL_GAS_RESILIENCE_MIN, CELL_GAS_RESILIENCE_MAX)
                self.perception = random.uniform(CELL_GAS_PERCEPTION_MIN, CELL_GAS_PERCEPTION_MAX)
                self.speed = random.uniform(CELL_GAS_SPEED_MIN, CELL_GAS_SPEED_MAX)
                self.lightEmission = random.uniform(CELL_GAS_LIGHTEMISSION_MIN, CELL_GAS_LIGHTEMISSION_MAX)
                self.lightAbsorption = random.uniform(CELL_GAS_LIGHTABSORPTION_MIN, CELL_GAS_LIGHTABSORPTION_MAX)
                self.inertEmission = random.uniform(CELL_GAS_INERTEMISSION_MIN, CELL_GAS_INERTEMISSION_MAX)
                self.inertAbsorption = random.uniform(CELL_GAS_INERTABSORPTION_MIN, CELL_GAS_INERTABSORPTION_MAX)
                self.mutationRate = random.uniform(CELL_GAS_MUTATIONRATE_MIN, CELL_GAS_MUTATIONRATE_MAX)
                self.lifeExpectancyMin = random.uniform(CELL_GAS_LIFEEXPECTANCYMIN_MIN, CELL_GAS_LIFEEXPECTANCYMIN_MAX)
                self.lifeExpectancyMax = random.uniform(CELL_GAS_LIFEEXPECTANCYMAX_MIN, CELL_GAS_LIFEEXPECTANCYMAX_MAX)
                self.fertilityRate = random.uniform(CELL_GAS_FERTILITYRATE_MIN, CELL_GAS_FERTILITYRATE_MAX)
                self.fertilityAgeMin = random.uniform(CELL_GAS_FERTILITYAGEMIN_MIN, CELL_GAS_FERTILITYAGEMIN_MAX)
                self.fertilityAgeMax = random.uniform(CELL_GAS_FERTILITYAGEMAX_MIN, CELL_GAS_FERTILITYAGEMAX_MAX)
                self.fertilityEnergy = random.uniform(CELL_GAS_FERTILITYENERGY_MIN, CELL_GAS_FERTILITYENERGY_MAX)
                self.mass = random.uniform(CELL_GAS_MASS_MIN, CELL_GAS_MASS_MAX)
                self.height = random.uniform(CELL_GAS_HEIGHT_MIN, CELL_GAS_HEIGHT_MAX)
                self.prefHeight = random.uniform(CELL_GAS_PREFHEIGHT_MIN, CELL_GAS_PREFHEIGHT_MAX)
                self.lightStorage = random.uniform (CELL_GAS_LIGHTSTORAGE_MIN, CELL_GAS_LIGHTSTORAGE_MAX)
                self.energyStorage = random.uniform(CELL_GAS_ENERGYSTORAGE_MIN, CELL_GAS_ENERGYSTORAGE_MAX)
                self.inertStorage = random.uniform(CELL_GAS_INERTSTORAGE_MIN, CELL_GAS_INERTSTORAGE_MAX)
                self.birthColor = random.uniform(CELL_GAS_COLOR_MIN, CELL_GAS_COLOR_MIN)
                self.color = self.birthColor
                self.setEnergy(random.uniform(CELL_GAS_ENERGY_MIN, CELL_GAS_ENERGY_MAX), "spawnNew GAS")
                # self.energy = random.uniform(CELL_GAS_ENERGY, CELL_GAS_ENERGY)
            case CellState.LIQUID:
                self.growthRate = random.uniform(CELL_LIQUID_GROWTHRATE_MIN, CELL_LIQUID_GROWTHRATE_MAX)
                self.resilience = random.uniform(CELL_LIQUID_RESILIENCE_MIN, CELL_LIQUID_RESILIENCE_MAX)
                self.perception = random.uniform(CELL_LIQUID_PERCEPTION_MIN, CELL_LIQUID_PERCEPTION_MAX)
                self.speed = random.uniform(CELL_LIQUID_SPEED_MIN, CELL_LIQUID_SPEED_MAX)
                self.lightEmission = random.uniform(CELL_LIQUID_LIGHTEMISSION_MIN, CELL_LIQUID_LIGHTEMISSION_MAX)
                self.lightAbsorption = random.uniform(CELL_LIQUID_LIGHTABSORPTION_MIN, CELL_LIQUID_LIGHTABSORPTION_MAX)
                self.inertEmission = random.uniform(CELL_LIQUID_INERTEMISSION_MIN, CELL_LIQUID_INERTEMISSION_MAX)
                self.inertAbsorption = random.uniform(CELL_LIQUID_INERTABSORPTION_MIN, CELL_LIQUID_INERTABSORPTION_MAX)
                self.mutationRate = random.uniform(CELL_LIQUID_MUTATIONRATE_MIN, CELL_LIQUID_MUTATIONRATE_MAX)
                self.lifeExpectancyMin = random.uniform(CELL_LIQUID_LIFEEXPECTANCYMIN_MIN, CELL_LIQUID_LIFEEXPECTANCYMIN_MAX)
                self.lifeExpectancyMax = random.uniform(CELL_LIQUID_LIFEEXPECTANCYMAX_MIN, CELL_LIQUID_LIFEEXPECTANCYMAX_MAX)
                self.fertilityRate = random.uniform(CELL_LIQUID_FERTILITYRATE_MIN, CELL_LIQUID_FERTILITYRATE_MAX)
                self.fertilityAgeMin = random.uniform(CELL_LIQUID_FERTILITYAGEMIN_MIN, CELL_LIQUID_FERTILITYAGEMIN_MAX)
                self.fertilityAgeMax = random.uniform(CELL_LIQUID_FERTILITYAGEMAX_MIN, CELL_LIQUID_FERTILITYAGEMAX_MAX)
                self.fertilityEnergy = random.uniform(CELL_LIQUID_FERTILITYENERGY_MIN, CELL_LIQUID_FERTILITYENERGY_MAX)
                self.mass = random.uniform(CELL_LIQUID_MASS_MIN, CELL_LIQUID_MASS_MAX)
                self.height = random.uniform(CELL_LIQUID_HEIGHT_MIN, CELL_LIQUID_HEIGHT_MAX)
                self.prefHeight = random.uniform(CELL_LIQUID_PREFHEIGHT_MIN, CELL_LIQUID_PREFHEIGHT_MAX)
                self.lightStorage = random.uniform (CELL_LIQUID_LIGHTSTORAGE_MIN, CELL_LIQUID_LIGHTSTORAGE_MAX)
                self.energyStorage = random.uniform(CELL_LIQUID_ENERGYSTORAGE_MIN, CELL_LIQUID_ENERGYSTORAGE_MAX)
                self.inertStorage = random.uniform(CELL_LIQUID_INERTSTORAGE_MIN, CELL_LIQUID_INERTSTORAGE_MAX)
                self.birthColor = random.uniform(CELL_LIQUID_COLOR_MIN, CELL_LIQUID_COLOR_MIN)
                self.color = self.birthColor
                self.setEnergy(random.uniform(CELL_LIQUID_ENERGY_MIN, CELL_LIQUID_ENERGY_MAX), "spawnNew LIQUID")
                # self.energy = random.uniform(CELL_LIQUID_ENERGY, CELL_LIQUID_ENERGY)
            case CellState.MESOPHASE:
                self.growthRate = random.uniform(CELL_MESOPHASE_GROWTHRATE_MIN, CELL_MESOPHASE_GROWTHRATE_MAX)
                self.resilience = random.uniform(CELL_MESOPHASE_RESILIENCE_MIN, CELL_MESOPHASE_RESILIENCE_MAX)
                self.perception = random.uniform(CELL_MESOPHASE_PERCEPTION_MIN, CELL_MESOPHASE_PERCEPTION_MAX)
                self.speed = random.uniform(CELL_MESOPHASE_SPEED_MIN, CELL_MESOPHASE_SPEED_MAX)
                self.lightEmission = random.uniform(CELL_MESOPHASE_LIGHTEMISSION_MIN, CELL_MESOPHASE_LIGHTEMISSION_MAX)
                self.lightAbsorption = random.uniform(CELL_MESOPHASE_LIGHTABSORPTION_MIN, CELL_MESOPHASE_LIGHTABSORPTION_MAX)
                self.inertEmission = random.uniform(CELL_MESOPHASE_INERTEMISSION_MIN, CELL_MESOPHASE_INERTEMISSION_MAX)
                self.inertAbsorption = random.uniform(CELL_MESOPHASE_INERTABSORPTION_MIN, CELL_MESOPHASE_INERTABSORPTION_MAX)
                self.mutationRate = random.uniform(CELL_MESOPHASE_MUTATIONRATE_MIN, CELL_MESOPHASE_MUTATIONRATE_MAX)
                self.lifeExpectancyMin = random.uniform(CELL_MESOPHASE_LIFEEXPECTANCYMIN_MIN, CELL_MESOPHASE_LIFEEXPECTANCYMIN_MAX)
                self.lifeExpectancyMax = random.uniform(CELL_MESOPHASE_LIFEEXPECTANCYMAX_MIN, CELL_MESOPHASE_LIFEEXPECTANCYMAX_MAX)
                self.fertilityRate = random.uniform(CELL_MESOPHASE_FERTILITYRATE_MIN, CELL_MESOPHASE_FERTILITYRATE_MAX)
                self.fertilityAgeMin = random.uniform(CELL_MESOPHASE_FERTILITYAGEMIN_MIN, CELL_MESOPHASE_FERTILITYAGEMIN_MAX)
                self.fertilityAgeMax = random.uniform(CELL_MESOPHASE_FERTILITYAGEMAX_MIN, CELL_MESOPHASE_FERTILITYAGEMAX_MAX)
                self.fertilityEnergy = random.uniform(CELL_MESOPHASE_FERTILITYENERGY_MIN, CELL_MESOPHASE_FERTILITYENERGY_MAX)
                self.mass = random.uniform(CELL_MESOPHASE_MASS_MIN, CELL_MESOPHASE_MASS_MAX)
                self.height = random.uniform(CELL_MESOPHASE_HEIGHT_MIN, CELL_MESOPHASE_HEIGHT_MAX)
                self.prefHeight = random.uniform(CELL_MESOPHASE_PREFHEIGHT_MIN, CELL_MESOPHASE_PREFHEIGHT_MAX)
                self.lightStorage = random.uniform (CELL_MESOPHASE_LIGHTSTORAGE_MIN, CELL_MESOPHASE_LIGHTSTORAGE_MAX)
                self.energyStorage = random.uniform(CELL_MESOPHASE_ENERGYSTORAGE_MIN, CELL_MESOPHASE_ENERGYSTORAGE_MAX)
                self.inertStorage = random.uniform(CELL_MESOPHASE_INERTSTORAGE_MIN, CELL_MESOPHASE_INERTSTORAGE_MAX)
                self.birthColor = random.uniform(CELL_MESOPHASE_COLOR_MIN, CELL_MESOPHASE_COLOR_MIN)
                self.color = self.birthColor
                self.setEnergy(random.uniform(CELL_MESOPHASE_ENERGY_MIN, CELL_MESOPHASE_ENERGY_MAX), "spawnNew MESOPHASE")
                # self.energy = random.uniform(CELL_MESOPHASE_ENERGY, CELL_MESOPHASE_ENERGY)
            case CellState.SOLID: # CRYSTALLINE
                self.growthRate = random.uniform(CELL_SOLID_GROWTHRATE_MIN, CELL_SOLID_GROWTHRATE_MAX)
                self.resilience = random.uniform(CELL_SOLID_RESILIENCE_MIN, CELL_SOLID_RESILIENCE_MAX)
                self.perception = random.uniform(CELL_SOLID_PERCEPTION_MIN, CELL_SOLID_PERCEPTION_MAX)
                self.speed = random.uniform(CELL_SOLID_SPEED_MIN, CELL_SOLID_SPEED_MAX)
                self.lightEmission = random.uniform(CELL_SOLID_LIGHTEMISSION_MIN, CELL_SOLID_LIGHTEMISSION_MAX)
                self.lightAbsorption = random.uniform(CELL_SOLID_LIGHTABSORPTION_MIN, CELL_SOLID_LIGHTABSORPTION_MAX)
                self.inertEmission = random.uniform(CELL_SOLID_INERTEMISSION_MIN, CELL_SOLID_INERTEMISSION_MAX)
                self.inertAbsorption = random.uniform(CELL_SOLID_INERTABSORPTION_MIN, CELL_SOLID_INERTABSORPTION_MAX)
                self.mutationRate = random.uniform(CELL_SOLID_MUTATIONRATE_MIN, CELL_SOLID_MUTATIONRATE_MAX)
                self.lifeExpectancyMin = random.uniform(CELL_SOLID_LIFEEXPECTANCYMIN_MIN, CELL_SOLID_LIFEEXPECTANCYMIN_MAX)
                self.lifeExpectancyMax = random.uniform(CELL_SOLID_LIFEEXPECTANCYMAX_MIN, CELL_SOLID_LIFEEXPECTANCYMAX_MAX)
                self.fertilityRate = random.uniform(CELL_SOLID_FERTILITYRATE_MIN, CELL_SOLID_FERTILITYRATE_MAX)
                self.fertilityAgeMin = random.uniform(CELL_SOLID_FERTILITYAGEMIN_MIN, CELL_SOLID_FERTILITYAGEMIN_MAX)
                self.fertilityAgeMax = random.uniform(CELL_SOLID_FERTILITYAGEMAX_MIN, CELL_SOLID_FERTILITYAGEMAX_MAX)
                self.fertilityEnergy = random.uniform(CELL_SOLID_FERTILITYENERGY_MIN, CELL_SOLID_FERTILITYENERGY_MAX)
                self.mass = random.uniform(CELL_SOLID_MASS_MIN, CELL_SOLID_MASS_MAX)
                self.height = random.uniform(CELL_SOLID_HEIGHT_MIN, CELL_SOLID_HEIGHT_MAX)
                self.prefHeight = random.uniform(CELL_SOLID_PREFHEIGHT_MIN, CELL_SOLID_PREFHEIGHT_MAX)
                self.lightStorage = random.uniform (CELL_SOLID_LIGHTSTORAGE_MIN, CELL_SOLID_LIGHTSTORAGE_MAX)
                self.energyStorage = random.uniform(CELL_SOLID_ENERGYSTORAGE_MIN, CELL_SOLID_ENERGYSTORAGE_MAX)
                self.inertStorage = random.uniform(CELL_SOLID_INERTSTORAGE_MIN, CELL_SOLID_INERTSTORAGE_MAX)
                self.birthColor = random.uniform(CELL_SOLID_COLOR_MIN, CELL_SOLID_COLOR_MIN)
                self.color = self.birthColor
                self.setEnergy(random.uniform(CELL_SOLID_ENERGY_MIN, CELL_SOLID_ENERGY_MAX), "spawnNew SOLID")
                # self.energy = random.uniform(CELL_SOLID_ENERGY, CELL_SOLID_ENERGY)
            case CellState.INERT:
                self.growthRate = random.uniform(CELL_INERT_GROWTHRATE_MIN, CELL_INERT_GROWTHRATE_MAX)
                self.resilience = random.uniform(CELL_INERT_RESILIENCE_MIN, CELL_INERT_RESILIENCE_MAX)
                self.perception = random.uniform(CELL_INERT_PERCEPTION_MIN, CELL_INERT_PERCEPTION_MAX)
                self.speed = random.uniform(CELL_INERT_SPEED_MIN, CELL_INERT_SPEED_MAX)
                self.lightEmission = random.uniform(CELL_INERT_LIGHTEMISSION_MIN, CELL_INERT_LIGHTEMISSION_MAX)
                self.lightAbsorption = random.uniform(CELL_INERT_LIGHTABSORPTION_MIN, CELL_INERT_LIGHTABSORPTION_MAX)
                self.inertEmission = random.uniform(CELL_INERT_INERTEMISSION_MIN, CELL_INERT_INERTEMISSION_MAX)
                self.inertAbsorption = random.uniform(CELL_INERT_INERTABSORPTION_MIN, CELL_INERT_INERTABSORPTION_MAX)
                self.mutationRate = random.uniform(CELL_INERT_MUTATIONRATE_MIN, CELL_INERT_MUTATIONRATE_MAX)
                self.lifeExpectancyMin = random.uniform(CELL_INERT_LIFEEXPECTANCYMIN_MIN, CELL_INERT_LIFEEXPECTANCYMIN_MAX)
                self.lifeExpectancyMax = random.uniform(CELL_INERT_LIFEEXPECTANCYMAX_MIN, CELL_INERT_LIFEEXPECTANCYMAX_MAX)
                self.fertilityRate = random.uniform(CELL_INERT_FERTILITYRATE_MIN, CELL_INERT_FERTILITYRATE_MAX)
                self.fertilityAgeMin = random.uniform(CELL_INERT_FERTILITYAGEMIN_MIN, CELL_INERT_FERTILITYAGEMIN_MAX)
                self.fertilityAgeMax = random.uniform(CELL_INERT_FERTILITYAGEMAX_MIN, CELL_INERT_FERTILITYAGEMAX_MAX)
                self.fertilityEnergy = random.uniform(CELL_INERT_FERTILITYENERGY_MIN, CELL_INERT_FERTILITYENERGY_MAX)
                self.mass = random.uniform(CELL_INERT_MASS_MIN, CELL_INERT_MASS_MAX)
                self.height = random.uniform(CELL_INERT_HEIGHT_MIN, CELL_INERT_HEIGHT_MAX)
                self.prefHeight = random.uniform(CELL_INERT_PREFHEIGHT_MIN, CELL_INERT_PREFHEIGHT_MAX)
                self.lightStorage = random.uniform (CELL_INERT_LIGHTSTORAGE_MIN, CELL_INERT_LIGHTSTORAGE_MAX)
                self.energyStorage = random.uniform(CELL_INERT_ENERGYSTORAGE_MIN, CELL_INERT_ENERGYSTORAGE_MAX)
                self.inertStorage = random.uniform(CELL_INERT_INERTSTORAGE_MIN, CELL_INERT_INERTSTORAGE_MAX)
                self.birthColor = random.uniform(CELL_INERT_COLOR_MIN, CELL_INERT_COLOR_MIN)
                self.color = self.birthColor
                self.setEnergy(random.uniform(CELL_INERT_ENERGY_MIN, CELL_INERT_ENERGY_MAX), "spawnNew INERT")
                # self.energy = random.uniform(CELL_INERT_ENERGY, CELL_INERT_ENERGY)
            case _:
                self.growthRate = random.uniform(CELL_BASE_GROWTHRATE_MIN, CELL_BASE_GROWTHRATE_MAX)
                self.resilience = random.uniform(CELL_BASE_RESILIENCE_MIN, CELL_BASE_RESILIENCE_MAX)
                self.perception = random.uniform(CELL_BASE_PERCEPTION_MIN, CELL_BASE_PERCEPTION_MAX)
                self.speed = random.uniform(CELL_BASE_SPEED_MIN, CELL_BASE_SPEED_MAX)
                self.lightEmission = random.uniform(CELL_BASE_LIGHTEMISSION_MIN, CELL_BASE_LIGHTEMISSION_MAX)
                self.lightAbsorption = random.uniform(CELL_BASE_LIGHTABSORPTION_MIN, CELL_BASE_LIGHTABSORPTION_MAX)
                self.inertEmission = random.uniform(CELL_BASE_INERTEMISSION_MIN, CELL_BASE_INERTEMISSION_MAX)
                self.inertAbsorption = random.uniform(CELL_BASE_INERTABSORPTION_MIN, CELL_BASE_INERTABSORPTION_MAX)
                self.mutationRate = random.uniform(CELL_BASE_MUTATIONRATE_MIN, CELL_BASE_MUTATIONRATE_MAX)
                self.lifeExpectancyMin = random.uniform(CELL_BASE_LIFEEXPECTANCYMIN_MIN, CELL_BASE_LIFEEXPECTANCYMIN_MAX)
                self.lifeExpectancyMax = random.uniform(CELL_BASE_LIFEEXPECTANCYMAX_MIN, CELL_BASE_LIFEEXPECTANCYMAX_MAX)
                self.fertilityRate = random.uniform(CELL_BASE_FERTILITYRATE_MIN, CELL_BASE_FERTILITYRATE_MAX)
                self.fertilityAgeMin = random.uniform(CELL_BASE_FERTILITYAGEMIN_MIN, CELL_BASE_FERTILITYAGEMIN_MAX)
                self.fertilityAgeMax = random.uniform(CELL_BASE_FERTILITYAGEMAX_MIN, CELL_BASE_FERTILITYAGEMAX_MAX)
                self.fertilityEnergy = random.uniform(CELL_BASE_FERTILITYENERGY_MIN, CELL_BASE_FERTILITYENERGY_MAX)
                self.mass = random.uniform(CELL_BASE_MASS_MIN, CELL_BASE_MASS_MAX)
                self.prefHeight = random.uniform(CELL_BASE_PREFHEIGHT_MIN, CELL_BASE_PREFHEIGHT_MAX)
                self.height = random.uniform(CELL_BASE_HEIGHT_MIN, CELL_BASE_HEIGHT_MAX)
                self.lightStorage = random.uniform (CELL_BASE_LIGHTSTORAGE_MIN, CELL_BASE_LIGHTSTORAGE_MAX)
                self.energyStorage = random.uniform(CELL_BASE_ENERGYSTORAGE_MIN, CELL_BASE_ENERGYSTORAGE_MAX)
                self.inertStorage = random.uniform(CELL_BASE_INERTSTORAGE_MIN, CELL_BASE_INERTSTORAGE_MAX)
                self.birthColor = random.uniform(CELL_BASE_COLOR_MIN, CELL_BASE_COLOR_MIN)
                self.color = self.birthColor
                self.setEnergy(random.uniform(CELL_BASE_ENERGY_MIN, CELL_BASE_ENERGY_MAX), "spawnNew BASE")
                # self.energy = random.uniform(CELL_BASE_ENERGY, CELL_BASE_ENERGY)

        if self.fertilityAgeMin > self.fertilityAgeMax:
            self.fertilityAgeMin, self.fertilityAgeMax = self.fertilityAgeMax, self.fertilityAgeMin
        if self.fertilityAgeMin < 5:
            self.fertilityAgeMin += 2

        if self.lifeExpectancyMin > self.lifeExpectancyMax:
            self.lifeExpectancyMin, self.lifeExpectancyMax = self.lifeExpectancyMax, self.lifeExpectancyMin
        if self.lifeExpectancyMin < 10:
            self.lifeExpectancyMin += 2

        self.lifeExpectancy = random.uniform(self.lifeExpectancyMin, self.lifeExpectancyMax)
    
    def spawnChild(self, parent):
        self.turnCount = parent.turnCount - 1 # Set it as eligible for a turn, i guess
        # self.energy = random.uniform(CELL_BASE_ENERGY_MIN, CELL_BASE_ENERGY_MAX)
        energyChild = max(0.001,min(1000,(((random.uniform(CELL_BASE_ENERGY_MIN, CELL_BASE_ENERGY_MAX))/2)+((parent.energy)/2)) * parent.mutationRate/50))
        self.setEnergy(energyChild, "spawnChild BASsE")

        if self.energy > (parent.energy*0.5):
            self.setEnergy((parent.energy*0.5), "spawnChild self.energy > (parent.energy*0.5)")
            # self.energy = (parent.energy*0.5)
        self.birthColor = self.parent.color
        self.color = self.birthColor
        self.phaseTransition()

        match self.state:
            case CellState.PLASMA:
                self.mutationRate = (random.uniform(CELL_PLASMA_MUTATIONRATE_MIN, CELL_PLASMA_MUTATIONRATE_MAX) + parent.mutationRate) / 2
            case CellState.GAS:
                self.mutationRate = (random.uniform(CELL_GAS_MUTATIONRATE_MIN, CELL_GAS_MUTATIONRATE_MAX) + parent.mutationRate) / 2
            case CellState.LIQUID:
                self.mutationRate = (random.uniform(CELL_LIQUID_MUTATIONRATE_MIN, CELL_LIQUID_MUTATIONRATE_MAX) + parent.mutationRate) / 2
            case CellState.MESOPHASE:
                self.mutationRate = (random.uniform(CELL_MESOPHASE_MUTATIONRATE_MIN, CELL_MESOPHASE_MUTATIONRATE_MAX) + parent.mutationRate) / 2
            case CellState.SOLID:
                self.mutationRate = (random.uniform(CELL_SOLID_MUTATIONRATE_MIN, CELL_SOLID_MUTATIONRATE_MAX) + parent.mutationRate) / 2
            case CellState.INERT:
                self.mutationRate = (random.uniform(CELL_INERT_MUTATIONRATE_MIN, CELL_INERT_MUTATIONRATE_MAX) + parent.mutationRate) / 2
            case _:
                self.mutationRate = (random.uniform(CELL_BASE_MUTATIONRATE_MIN, CELL_BASE_MUTATIONRATE_MAX) + parent.mutationRate) / 2
                self.stats.addCellStateChange("???")

        self.mutateProp(self.generalStatsList+["luck"])
        self.normalizeProps(self.generalStatsList)

        self.role = random.choice([parent.role, random.choice(CELL_ROLES)])
        self.lifeExpectancy = random.uniform(self.lifeExpectancyMin, self.lifeExpectancyMax)

        self.luck += (parent.luck/(80+parent.age)) * self.mutationRate # Get a bit of luck from your parent
        self.height = self.height/2 # they DO also wanna mutate, cause mutate then lost 50% parent height
        self.mass = self.mass/2

        if self.fertilityAgeMin > self.fertilityAgeMax:
            self.fertilityAgeMin, self.fertilityAgeMax = self.fertilityAgeMax, self.fertilityAgeMin
        if self.fertilityAgeMin < 5:
            self.fertilityAgeMin += 2

        if self.lifeExpectancyMin > self.lifeExpectancyMax:
            self.lifeExpectancyMin, self.lifeExpectancyMax = self.lifeExpectancyMax, self.lifeExpectancyMin
        if self.lifeExpectancyMin < 10:
            self.lifeExpectancyMin += 2

        self.lifeExpectancy = random.uniform(self.lifeExpectancyMin, self.lifeExpectancyMax)

    def moveOrSquish(self, moving, direction):
        if self.energy < 0:
            self.setEnergy(0, "moveOrSquish: Not enough energy to move")
            self.alive = False
            self.environment.removeCellFromGrid(self)
            self.visible = False
            #print(f"Died from cuddles. Energy: {self.energy}, lost {squishEnergyTransfer} this turn")
            self.memory.append((self.turnCount, "Died from lazy cuddle"))
            self.stats.addCellDeath(CELL_DEATH_REASON_SQUISH)
            return
        dx, dy = direction
        new_x = (self.x + dx) % self.environment.grid.shape[0]
        new_y = (self.y + dy) % self.environment.grid.shape[1]
        # Use signalGrid for perception-based movement
        signal_at_target = self.environment.signalGrid[new_x, new_y]
        #print(f"moveOrSquish called by Cell {self.id} targeting ({new_x}, {new_y}), signal: {signal_at_target}")  # Debug

        # Found an empty space
        # PUSH
        if self.environment.canAddCellAt(new_x, new_y):
            self.environment.moveCellTo(new_x, new_y, self)
            #print(f"Escaped a death squish! Ran to signal {signal_at_target}")
            self.stats.addCellDeathEscape()
            self.stats.addCellMove()
            self.luck += 1
            self.fertilityRate -= self.luckChoice() * self.fertilityRate/100
            self.resilience += self.luckChoice() * (self.resilience/100)
            if self.energy > 0:
                self.setEnergy(self.energy - (self.turnRollAlt * (self.energy/2)), "moveOrSquish: escaped squish")
                if self.energy < 0:
                    self.setEnergy(0, "moveOrSquish: escaped squish but died :(")
                # self.energy -= self.turnRollAlt * self.energy
            self.memory.append((self.turnCount, "Escaped a death squish!", signal_at_target))
            if not self.alive: # if cell is already inert and needs to move, update inertGrid
                self.environment.addInertAt(self.x, self.y, (self.tightTurnRoll * CELL_DEATH_RELEASE_INERT))
            return True

        cell = self.environment.getCellAt(new_x, new_y)
        if isinstance(cell, Cell):
            #if cell.speed < (self.speed):
            #print(f"Cell {self.id} collided with Cell {cell.id} at ({new_x}, {new_y})")  # Debug

            if cell.resilience > (self.resilience):
                #print(f"Cell {self.id} squished by Cell {cell.id}")  # Debug
                # the target cell get squished
                ratio = random.uniform(CELL_DEATH_RELEASE_SQUISH_MIN, CELL_DEATH_RELEASE_SQUISH_MAX)
                squishEnergyTransfer = self.energy * ratio/4 # Squish release of energy (norty?!)
                cell.setEnergy(cell.energy + squishEnergyTransfer, "moveOrSquish: transfer squish to next cell in chain")
                # cell.energy += squishEnergyTransfer
                moving.setEnergy(moving.energy + squishEnergyTransfer, "moveOrSquish: transfer squish to moving cell")
                # moving.energy += squishEnergyTransfer
                self.alive = False
                self.environment.removeCellFromGrid(self)
                self.visible = False
                #print(f"Died from cuddles. Energy: {self.energy}, lost {squishEnergyTransfer} this turn")
                self.memory.append((self.turnCount, "Died from cuddles", squishEnergyTransfer))
                self.stats.addCellDeath(CELL_DEATH_REASON_SQUISH)
                self.environment.addInertAt(self.x, self.y, (random.uniform(CELL_DEATH_RELEASE_SQUISH_MIN, CELL_DEATH_RELEASE_SQUISH_MIN)))
                self.stats.addCellDisintegrationDeath()
                return False
            
            else:
                # The destination cell attempts to move away
                # DOUBLE++ PUSH
                cell.moveOrSquish(self, direction)
                if self.environment.canAddCellAt(new_x, new_y):
                    self.environment.moveCellTo(new_x, new_y, self)
                    #print(f"The Vengabus is Evolving O.o at signal {signal_at_target}")
                    self.stats.addCellPush()
                    self.stats.addCellMove()
                    self.memory.append((self.turnCount, f"The Vengabus is Evolving O.o at signal {signal_at_target} (Move Bounced)", (new_x, new_y)))
                    self.luck += 2
                    if not self.alive:
                        self.environment.addInertAt(self.x, self.y, CELL_DEATH_RELEASE_INERT)
                return True
                    
        print(f"IDK WHAT'S HERE: {cell} at signal {signal_at_target}")
        self.memory.append((self.turnCount, f"IDK WHAT'S HERE: {cell} at signal {signal_at_target}", signal_at_target))
        return True
        
    def move(self):
        if not self.alive or self.energy < CELL_MOVE_ENERGY_MIN:
            self.memory.append((self.turnCount, "had a lie in today", self.energy))
            self.stats.addCellStop()
            # print(f"Not moving because alive is {self.alive} & energy is {self.energy}")
            return
        
        # Movement based on environmental signals and nutrient concentration
        potentialMoves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(potentialMoves)
        blockCounter = 0
        maxMoveAttempts = 4

        while blockCounter < maxMoveAttempts:
            bestMove = None
            maxSignal = -1
            #print(f"Cell {self.id} evaluating moves at ({self.x}, {self.y})")  # Debug
            if self.perception < CELL_BLINDLESS_LEVEL:
                dx, dy = random.choice([-1,0,1]), random.choice([-1,0,1])
                new_x = (self.x + dx) % self.environment.grid.shape[0]
                new_y = (self.y + dy) % self.environment.grid.shape[1]
            else:
                for dx, dy in potentialMoves:
                    new_x = (self.x + dx) % self.environment.grid.shape[0]
                    new_y = (self.y + dy) % self.environment.grid.shape[1]
                    signal = self.environment.signalGrid[new_x, new_y]
                    self.memory.append((self.turnCount, "Considered another direction", (dx, dy)))
                    #print(f"  Checking move to ({new_x}, {new_y}), signal: {signal}")  # Debug
                    if signal > maxSignal and self.environment.canAddCellAt(new_x, new_y):
                        # print(f"Best signal")
                        bestMove = (dx), (dy)
                        maxSignal = signal
                

            if bestMove:
                (dx), (dy) = bestMove
                new_x = (self.x + dx) % self.environment.grid.shape[0]
                new_y = (self.y + dy) % self.environment.grid.shape[1]
                if new_x == self.prevX and new_y == self.prevY:
                    self.moveLoopCounter += 1
                    #print({self.moveLoopCounter})
                if self.moveLoopCounter > 3:
                    (dx), (dy) = random.choice(potentialMoves)
                    #print(f"reset move loop counter")
                    self.moveLoopCounter = 0
                    new_x = (self.x + dx) % self.environment.grid.shape[0]
                    new_y = (self.y + dy) % self.environment.grid.shape[1]

                #print(f"Cell {self.id} moving to ({new_x}, {new_y}) with signal {maxSignal}")  # Debug
            if self.environment.canAddCellAt(abs(new_x), abs(new_y)):
                self.prevX = (self.x)
                self.prevY = (self.y)
                self.environment.moveCellTo(abs(new_x), abs(new_y), self)
                self.environment.depleteInertAt(self.x,self.y,(self.environment.getInertAt(self.x, self.y)/100))
                self.waterErosion(dx, dy)
                self.stats.addCellMove()
                self.memory.append((self.turnCount, f"Moved to signal {maxSignal}", (new_x, new_y)))
                break
            
            blockCounter += 1
            self.luck -= 0.2
            #print(f"Cell {self.id} blocked. Attempt {blockCounter}/{maxMoveAttempts}")  # Debug
            self.memory.append((self.turnCount, f"You're really gonna block me {blockCounter} time(s)?", (self.x, self.y)))

            #print(f"Grid at ({new_x}, {new_y}): {type(self.environment.grid[new_x, new_y])}, value: {self.environment.grid[new_x, new_y]}")
            if isinstance(self.environment.grid[new_x, new_y], Cell):
                target_cell = self.environment.grid[new_x, new_y]
                if self.resilience > target_cell.resilience:
                    # Current cell has higher resilience, attempt to push the target away
                    self.memory.append((self.turnCount, "Pushed weaker cell", (new_x, new_y)))
                    target_cell.moveOrSquish(self, (dx, dy))
                    self.luck += 0.1
                    return
            else:
                # Target cell is stronger
                self.memory.append((self.turnCount, "Blocked by stronger cell", (new_x, new_y)))
        else:
            # Handle non-cell cases (e.g., empty space, gas, or other markers)
            self.memory.append((self.turnCount, f"No cell at ({new_x}, {new_y}) to compare resilience", (self.x, self.y)))
            self.luck += 0.1
        
        if self.x != new_x or self.y != new_y:
            #print(f"Failed moving {self.id} from ({self.x}, {self.y}) to ({new_x}, {new_y})")
            self.memory.append((self.turnCount, "Move Failed", (new_x, new_y)))
                    
        else:
            #print(f"Failed moving {self.id} onto itself")
            self.memory.append((self.turnCount, "Move Failed", (new_x, new_y)))

    def waterErosion(self, dx, dy):
        if self.state == CellState.LIQUID:
            self.environment.depleteInertAt(self.x,self.y,(self.inertUnderCell/20))
            leftX = (self.x - dx) + (1 * abs(dy))
            leftY = (self.y - dy) + (1 * abs(dx))
            rightX = (self.x - dx) - (1 * abs(dy))
            rightY = (self.y - dy) - (1 * abs(dx))
            self.environment.addInertAt(leftX,leftY,(self.inertUnderCell/10))
            self.environment.addInertAt(rightX,rightY,(self.inertUnderCell/10))

    def getCellColor(self):
        if self.organism and self.organism.name:  # Sentient organisms
            if not hasattr(self.organism, "color"):
                self.organism.color = hsv_to_rgb((random.random(), self.turnRoll, self.turnRollAlt))
            return self.organism.color
        if not self.alive:  # Dead cells
            return hsv_to_rgb((self.color, 0.5, 0.2))  # Lower brightness for dead cells
        # Alive cells
        brightness = max(0.3, min(1.0, self.energy / self.topEnergy))  # Avoid excessive clamping
        return hsv_to_rgb((self.color, 0.8, brightness))

    def absorbNutrients(self):
        if self.alive:
            envLightLevel = self.environment.getLightAt(self.x, self.y) #lightGrid[self.x, self.y]
            lightAbsorbed = (envLightLevel/100) * self.lightAbsorption
            if (lightAbsorbed + self.energy) > self.energyStorage:
                self.environment.depleteLightAt(self.x, self.y, (lightAbsorbed * ENVIRONMENT_LIGHTABSORPTION_WASTE)/100)
                self.memory.append((self.turnCount, "Energy Reserves Full", lightAbsorbed))
                self.luck = self.luckChoice() * 0.1
                return
            else:
                self.setEnergy(self.energy + lightAbsorbed, "absorbNutrients: absorbed light")
                # self.energy += lightAbsorbed
                self.environment.depleteLightAt(self.x, self.y, (lightAbsorbed * ENVIRONMENT_LIGHTABSORPTION_WASTE))
                self.memory.append((self.turnCount, "Gained Light Energy", lightAbsorbed))
                #print(f"Turn {self.turnCount}: Cell {self.id} gained {lightAbsorbed} energy. Total: {self.energy}")
                # self.environment.lightGrid[self.x, self.y] = max(self.environment.lightGrid[self.x, self.y] - 0.02, 0)  # Deplete nutrients

    def emitLight(self):
        if self.state == CellState.PLASMA: # Plasma cells consistently emit high light
            self.lightEmission += self.luckChance() * (self.lightEmission/50)
            self.setEnergy(self.energy - (self.lightEmission), "emitLight: PLASMA")
            # self.energy -= self.lightEmission
            self.memory.append((self.turnCount, "Emitted light", (self.lightEmission)))
        elif random.random() < 0.01 and self.energy > self.fertilityEnergy: # Non-plasma cells have a random chance to emit light
            self.lightEmission += self.luckChance() * (self.lightEmission/100)
            self.setEnergy(self.energy - (self.lightEmission), "emitLight: random non PLASMA")
            # self.energy -= self.lightEmission
            self.luck += 1
            self.memory.append((self.turnCount, "Suddenly emitted light?!", self.lightEmission))
        self.environment.addLightAt(self.x, self.y, self.lightEmission)
            
    def waifuSignal(self):
        if self.alive:
            self.environment.addAttractivenessAt(self.x, self.y, ((self.attractiveness+100)/2))
            # waifuGrid[self.x, self.y] = min(waifuGrid[self.x, self.y] + self.attractiveness, 100)
            vibes = self.environment.getAttractivenessAt(self.x, self.y) # waifuGrid[self.x, self.y]
            self.fertilityRate += self.luckChoice() * (self.turnRoll * (self.fertilityRate/100))
            self.memory.append((self.turnCount, "DAT ASS SUCH FERTILE", vibes))
        else:
            self.environment.setAttractivenessAt(self.x, self.y, 0)
            # waifuGrid[self.x, self.y] = 0
            self.memory.append((self.turnCount, "Even my death turns people off!?", 0))
            
    # State of the cell: solid, liquid, gas, plasma, inert
    def phaseTransition(self):
        if self.energy > CELL_PLASMA_ENERGY_MIN:
            if not hasattr(self, "state") or self.state != CellState.PLASMA:
                self.state = CellState.PLASMA
                self.stats.addCellStateChange(CellState.PLASMA)
                self.color = ((random.uniform(CELL_PLASMA_COLOR_MIN, CELL_PLASMA_COLOR_MIN))+self.birthColor+self.color)/3
                self.memory.append((self.turnCount, "Became Plasma"))
            else:
                self.stats.addCellStateStable()
                self.memory.append((self.turnCount, "Still Plasma"))
        elif CELL_GAS_ENERGY_MIN < self.energy <= CELL_PLASMA_ENERGY_MIN:
            if not hasattr(self, "state") or self.state != CellState.GAS:
                self.state = CellState.GAS
                self.stats.addCellStateChange(CellState.GAS)
                self.color = ((random.uniform(CELL_GAS_COLOR_MIN, CELL_GAS_COLOR_MIN))+self.birthColor+self.color)/3
                self.memory.append((self.turnCount, "Became Gas"))
            else:
                self.stats.addCellStateStable()
                self.memory.append((self.turnCount, "Still Gas"))
        elif CELL_LIQUID_ENERGY_MIN < self.energy <= CELL_GAS_ENERGY_MIN:
            if not hasattr(self, "state") or self.state != CellState.LIQUID:
                self.state = CellState.LIQUID
                self.stats.addCellStateChange(CellState.LIQUID)
                self.color = ((random.uniform(CELL_LIQUID_COLOR_MIN, CELL_LIQUID_COLOR_MIN))+self.birthColor+self.color)/3
                self.memory.append((self.turnCount, "Became Liquid", 0))
            else:
                self.stats.addCellStateStable()
                self.memory.append((self.turnCount, "Still Liquid"))
        elif CELL_MESOPHASE_ENERGY_MIN < self.energy <= CELL_LIQUID_ENERGY_MIN:
            if not hasattr(self, "state") or self.state != CellState.MESOPHASE:
                self.state = CellState.MESOPHASE
                self.stats.addCellStateChange(CellState.MESOPHASE)
                self.color = ((random.uniform(CELL_MESOPHASE_COLOR_MIN, CELL_MESOPHASE_COLOR_MIN))+self.birthColor+self.color)/3
                self.memory.append((self.turnCount, "Entered the Mesophase", 0))
            else:
                self.stats.addCellStateStable()
                self.memory.append((self.turnCount, "Still Mesophase"))
        elif CELL_SOLID_ENERGY_MIN < self.energy <= CELL_MESOPHASE_ENERGY_MIN:
            if not hasattr(self, "state") or self.state != CellState.SOLID:
                self.state = CellState.SOLID
                self.stats.addCellStateChange(CellState.SOLID)
                self.color = ((random.uniform(CELL_SOLID_COLOR_MIN, CELL_SOLID_COLOR_MIN))+self.birthColor+self.color)/3
                self.memory.append((self.turnCount, "Got Hard", 0))
            else:
                self.stats.addCellStateStable()
                self.memory.append((self.turnCount, "Still Hard"))
        elif CELL_INERT_ENERGY_MIN < self.energy <= CELL_SOLID_ENERGY_MIN:
            if not hasattr(self, "state") or self.state != CellState.INERT:
                self.state = CellState.INERT
                self.stats.addCellStateChange(CellState.INERT)
                self.color = ((random.uniform(CELL_INERT_COLOR_MIN, CELL_INERT_COLOR_MIN))+self.birthColor+self.color)/3
                self.memory.append((self.turnCount, "Became Inert", 0))
            else:
                self.stats.addCellStateStable()
                self.memory.append((self.turnCount, "Still Inert"))
        else:
            if not hasattr(self, "state"):
                raise Exception("Cell {self.id} spawned from {cell.parent} with energy {self.energy}! Unable to set state in phaseTransition")

    def reproduce(self):
        if not self.alive:
            return False
        if self.age < self.fertilityAgeMin:
            self.stats.addCellYouth()
            self.mass += self.luckChoice() * self.mass/self.fertilityAgeMin
            self.memory.append((self.turnCount, "I'm just a kid!"))
            return False
        if self.luck > 100:
            self.luck = self.turnRollAlt * self.luck/2
            x, y = (self.x + random.choice([-1, 1])) % self.environment.grid.shape[0], (self.y + random.choice([-1, 1])) % self.environment.grid.shape[1]
            if self.turnRoll > 1 and self.environment.canAddCellAt(x, y):
                reproductionCost = (self.energy-(self.energy/CELL_REPRODUCTION_SUCCESS_COST))
                self.fertilityEnergy -= reproductionCost/6
                self.setEnergy(reproductionCost, "reproduce: super lucky")
                baby_cell = Cell(x, y, self.stats, self.environment, organismCheck=self.organism, parent=self)
                self.environment.setCellAt(x, y, baby_cell)
                self.memory.append((self.turnCount, "Okay, uh, what the fuck?! WHo fOrGOT TheIR KID!?!!", self.fertilityRate))
                self.stats.addCellBaby("WTF?!")
                return True
        if self.age > self.fertilityAgeMax:
            self.stats.addCellElderly()
            self.resilience -= self.resilience/100
            return False
        self.stats.addCellAdult()
        if self.energy < self.fertilityEnergy:
            self.stats.addCellBabyFailed("Exhausted")
            self.memory.append((self.turnCount, "Too lazy to fuck"))
            self.fertilityRate += self.fertilityRate/100 + 0.9
            self.energyStorage += self.luckChoice()
            return False
        if self.fertilityEnergy > 5:
            self.fertilityEnergy += self.fertilityEnergy/5

        if (self.turnRollAlt * 100) < self.fertilityRate or (self.attractiveness > ((self.CellAttractivenessTopRecord/10)*9)):
            # TODO: Factor in Luck
            else: 
            x, y = (self.x + random.choice([-1, 1])) % self.environment.grid.shape[0], (self.y + random.choice([-1, 1])) % self.environment.grid.shape[1]
            if self.environment.canAddCellAt(x, y):  # Empty spot
                reproductionCost = (self.energy/CELL_REPRODUCTION_SUCCESS_COST)
                self.fertilityEnergy += self.fertilityEnergy/100
                self.setEnergy(reproductionCost, "reproduce: Has Baby")
                self.energyStorage = self.energyStorage - reproductionCost/5
                baby_cell = Cell(x, y, self.stats, self.environment, organismCheck=self.organism, parent=self)
                self.environment.setCellAt(x, y, baby_cell)
                # print("UNEBEBEEEEEEEEEEEEEEEEE!!!!!!!!!!!!!!!!!!1!!!!!!!!!!!!!1!!!")
                if self.attractiveness < ((self.CellAttractivenessTopRecord/10)*9):
                    self.memory.append((self.turnCount, "Wait, une bebe?! Where did this thing come from!?", self.fertilityRate))
                    self.fertilityRate += self.luckChoice() * (self.turnRollAlt * (self.fertilityRate/100))
                    self.stats.addCellBaby("Fertile")
                else:
                    self.memory.append((self.turnCount, "Can't believe i'm finally a parent!", self.attractiveness))
                    self.stats.addCellBaby("Attractive")
                    self.fertilityRate += self.luckChoice() * (self.turnRoll * (self.fertilityRate/100))
                return True
            else:
                reproductionFailureCost = (self.energy/CELL_REPRODUCTION_FAILURE_COST)
                self.setEnergy(reproductionFailureCost, "reproduce: Baby having failed")
                # self.energy = reproductionFailureCost
                self.energyStorage = self.energyStorage + (self.luckChoice() * reproductionFailureCost/2)
                # print("Tried to UNEBEBEBEBEBEBEBEE BUT NO SPACE LEFT")
                self.stats.addCellBabyFailed("Overpopulation")
                self.memory.append((self.turnCount, "Didn't have room for even 1 bebe :("))
                self.fertilityRate += self.fertilityRate/100
                return False
        return False

    def disintegration(self):
        if self.alive = False:
            resilienceDecay = (self.resilience*(INERT_STONE_SOFTNESS/100))
            massDecay = (self.mass*(INERT_STONE_SOFTNESS/100))
            self.resilience -= resilienceDecay
            self.mass -= massDecay
            self.environment.addInertAt(self.x, self.y, massDecay)
            if self.state == CellState.INERT: # inert cells 'birth' enrichment onto environment
                enrichInert = self.mass/100 + 1
                self.mass -= enrichInert
                self.environment.addInertAt(self.x, self.y, (enrichInert * 0.2)) # 20% at self, 10% at each adjacent, 40% lost
                self.environment.addInertAt(self.x + 1, self.y, (enrichInert * 0.1)) # Top
                self.environment.addInertAt(self.x, self.y + 1, (enrichInert * 0.1)) # Right
                self.environment.addInertAt(self.x - 1, self.y, (enrichInert * 0.1)) # Bottom
                self.environment.addInertAt(self.x, self.y - 1, (enrichInert * 0.1)) # Left
                self.stats.addCellDisintegration()
                self.memory.append((self.turnCount, "Enriched the earth", (enrichInert * 0.6)))
                if self.mass <= 0:
                    # disappear from board
                    self.mass = 0
                    self.environment.removeCellFromGrid(self)
                    self.visible = False
                    self.alive = False
                    self.stats.addCellDisintegrationDeath()
                    self.memory.append((self.turnCount, "Oop bye"))

    def decay(self):
        if not self.alive:
            return
        self.setEnergy(self.energy - ((self.energy/100) - 1), "decay: turn decay")
        # self.energy -= (self.energy/100) - 1
        self.age += self.turnRollAlt * CELL_DECAY_AGE_PER_TURN
        self.growthRate += self.turnRoll
        self.height += self.growthRate/100
        self.lifeExpectancy += self.luckChoice()*(random.uniform(self.lifeExpectancyMin, self.lifeExpectancyMax)/100)
        self.attractiveness = self.turnRoll * (((self.energy*CELL_ATTRACTIVENESS_NORM_ENERGY)+ \
                                                            (self.age*CELL_ATTRACTIVENESS_NORM_AGE)+ \
                                                            (self.growthRate*CELL_ATTRACTIVENESS_NORM_GROWTH)+ \
                                                            (self.resilience*CELL_ATTRACTIVENESS_NORM_RESILIENCE)+ \
                                                            (self.perception*CELL_ATTRACTIVENESS_NORM_STRENGTH)+ \
                                                            (self.speed*CELL_ATTRACTIVENESS_NORM_SPEED)+ \
                                                            (self.lightEmission*CELL_ATTRACTIVENESS_NORM_LIGHTEMISSION) + \
                                                            (self.mutationRate*CELL_ATTRACTIVENESS_NORM_MUTATIONRATE) + \
                                                            (self.lifeExpectancy*CELL_ATTRACTIVENESS_NORM_LIFE_EXPECTANCY) + \
                                                            (self.mass*CELL_ATTRACTIVENESS_NORM_MASS) + \
                                                            (self.height*CELL_ATTRACTIVENESS_NORM_HEIGHT)) / \
                                                            (11*CELL_ATTRACTIVENESS_NORM_NORM))
        if self.attractiveness > self.CellAttractivenessTopRecord:
            self.CellAttractivenessTopRecord = self.attractiveness
            self.lifeExpectancyMax += self.lifeExpectancyMax/100
        if self.energy > self.energyStorage:
            self.setEnergy(self.energyStorage, "decay: energy > storage cap")
            # self.energy = self.energyStorage
            self.energyStorage += 1
            self.energy -= self.energy/100
        if self.energy > self.cellEnergyRecord:
            self.cellEnergyRecord = self.energy
            self.topEnergyDecay = self.tightTurnRoll * (self.energy/100)
            if self.energy > self.topEnergyDecay:
                self.setEnergy(self.energy - (self.topEnergyDecay), "decay: too much energy bonus decay")
                # self.energy -= self.topEnergyDecay
                self.memory.append((self.turnCount, "Fuck, being this cool is too hard, I lost energy", self.topEnergyDecay))
            else:
                self.setEnergy(0, "decay: can't decay")
                # self.energy = 0
        #print(f"Rated {self.attractiveness}% hot")
        self.memory.append((self.turnCount, f"I'm really rated {self.attractiveness} percent hot!?", self.attractiveness))
        if (self.energy <= 0) or (self.age >= (self.turnRollAlt * self.lifeExpectancy)):  # Death by starvation or old age
            self.alive = False
            #print(f"Died from state {self.state} Energy: {self.energy}, lost {(1 / self.resilience) * self.speed} this turn")
            if self.age < self.lifeExpectancy:
                self.stats.addCellDeath(CELL_DEATH_REASON_STARVATION)
                self.memory.append((self.turnCount, "I got too tired", {"energy": self.energy, "age": self.age, "lifeExpectancy": self.lifeExpectancy}))
            else:
                self.stats.addCellDeath(CELL_DEATH_REASON_AGE)
                self.memory.append((self.turnCount, "I got too old", {"energy": self.energy, "age": self.age, "lifeExpectancy": self.lifeExpectancy}))
            #else:
            #    self.stats.addCellDeath(CELL_DEATH_REASON_STARVATION)
            SimulationRecorder().recordDeath(self)
            self.state = CellState.INERT
            self.mass = ((self.mass*2)+self.energy)
            resilienceDecay = (self.resilience*(INERT_STONE_SOFTNESS/100))
            massDecay = (self.mass*(INERT_STONE_SOFTNESS/100))
            self.resilience -= resilienceDecay
            self.mass -= massDecay
            self.environment.addInertAt(self.x, self.y, massDecay)
            if self.lightEmission > 0 and self.energy > self.lightEmission: # releasing light storage on death
                self.environment.addLightAt(self.x, self.y, self.lightEmission)
                self.energy -= self.lightEmission/2
            self.setEnergy(0, "decay: after death setting 0")

    def normalizeProps(self, props):
            cellState = self.state.value.upper() #getattr(cell, CellState)
            for propName in props:
                cellPropVal = getattr(self, propName) # get the stats of the cell
                stateNameMax = f"CELL_{cellState}_{propName.upper()}_MAX"
                stateNameMin = f"CELL_{cellState}_{propName.upper()}_MIN"
                stateMaxVal = globals()[stateNameMax] # get the base stats of an average cell of it's state
                stateMinVal = globals()[stateNameMin]
                # print(f"check")
                if cellPropVal <= 0:
                    cellPropVal = 0
                    # print(f"{self.id} zeroing {getattr(self, propName)}")
                elif cellPropVal > stateMaxVal*5:
                    #print(f"{self.id} cheating pre {getattr(self, propName)}")
                    cellPropVal = (cellPropVal + (stateMaxVal*4)) / 10
                elif cellPropVal < stateMinVal/5:
                    cellPropVal = (cellPropVal + stateMinVal)
                elif cellPropVal > stateMaxVal and self.turnRoll > 1:
                    # print(f"{self.id} norm down pre {getattr(self, propName)}")
                    cellPropVal = (cellPropVal + stateMaxVal) / 2
                elif cellPropVal < stateMinVal and self.turnRollAlt > 1:
                    # print(f"{self.id} norm up pre {getattr(self, propName)}")
                    cellPropVal = (cellPropVal + stateMinVal) / 2
                setattr(self, propName, cellPropVal)
                # print(f"{self.id} post normie {getattr(self, propName)}")

    def setEnergy(self, newEnergy, readout):
        if hasattr(self, "energy"):
            NRG_READOUT_FILE.write(f"Cell {self.id} Current Energy is {self.energy}. New energy is {newEnergy}. Readout {readout} \n")
        else:
            NRG_READOUT_FILE.write(f"Cell {self.id}. Current Energy is None (setting first time). New energy is {newEnergy}. Readout {readout} \n")
        self.energy = newEnergy

    def needTurn(self, turn):
        return self.turnCount < turn
    
    def mutateProp(self, props):
            for propName in props:
                parentVal = getattr(self.parent, propName)
                mutatedValue = parentVal + (self.luckChoice() * ((parentVal/100) * self.mutationRate))
                setattr(self, propName, mutatedValue)

    def luckChoice(self):
        return random.choices([-1, 1], k = 1, weights = [((self.luck + 100)/200), (1-((self.luck + 100)/200))])[0] # Luck (assuming scaled -100 to 100) and a random chance weight the + or - choice

    def summarizeMemory(self):
        if self.alive == True:
            self.previousAlive = True
            return
        if self.alive == False and self.previousAlive == True:
            self.previousAlive = False

            onDeathStats = (f"\n Hey, Cell {self.id} here. Just passing on my memoir... Died at: {self.age}, on turn {self.turnCount}, at {self.x},{self.y}. Cell role: {self.role}. Attractiveness: {self.attractiveness}. Growth Decay Rate: {self.growthDecayRate}. Luck: {self.luck}. Highest Energy: {self.cellEnergyRecord}. Energy: {self.energy}. Growth Rate: {self.growthRate}. Resilience: {self.resilience}. Perception Strength: {self.perception}. Speed: {self.speed}. Light Emission: {self.lightEmission}. Light Absorption: {self.lightAbsorption}. Mutation Rate: {self.mutationRate}. Life Expectancy: {self.lifeExpectancy}. Fertility Rate: {self.fertilityRate}. Fertility Age: {self.fertilityAgeMin} - {self.fertilityAgeMax}. Energy needed for reproduction: {self.fertilityEnergy}. Mass: {self.mass}. Height: {self.height}. color: {self.color}.")
            
            #with open("birthDeathStats.txt", "a") as file: # a is append! :)
                #file.write(onDeathStats + "\n")

            #print(f"Cell {self.id} death written to birthDeathStats.txt successfully!")

    def checkBoundaries(self, props):
        for propName in props:
            cellPropVal = getattr(self, propName) # get the stats of the cell
            if cellPropVal <= 0:
                cellPropVal += self.turnRollAlt * 2
                setattr(self, propName, cellPropVal)

    def runLoop(self, turn):
        self.turnCount = turn
        self.getTurnInfo()
        self.move()
        self.absorbNutrients()
        self.phaseTransition()
        self.reproduce()
        self.disintegration()
        self.decay()
        self.waifuSignal()
        self.summarizeMemory()
        self.checkBoundaries(self.generalStatsList)
        if self.energy > self.topEnergy:
            self.topEnergy = self.energy