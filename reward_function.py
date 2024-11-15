def calculate_reward(state):
    # Initialize reward
    R = 0

    # a. Victory Points (VP)
    w_VP = 100
    R_VP = w_VP * state.victory_points
    R += R_VP

    # b. Tile Value
    w_Tile = 10
    tile_value = 0
    for settlement in state.settlements:
        tile_value += get_tile_value(settlement.tile)
    for city in state.cities:
        tile_value += 2 * get_tile_value(city.tile)  # Cities double resource production
    R_Tile = w_Tile * tile_value
    R += R_Tile

    # c. Hand Composition (Resource Cards)
    w_Hand = 5
    resource_diversity = len(state.hand.resources)
    resource_balance = -max(state.hand.resources.values())  # Penalize imbalance
    R_Hand = w_Hand * (resource_diversity + resource_balance)
    R += R_Hand

    # d. Robber Position
    w_Robber = 20
    if state.robber_position in state.player_tiles:
        harm = calculate_harm(state.robber_position, state)
    else:
        benefit = calculate_benefit(state.robber_position, state.opponents)
    R_Robber = w_Robber * (benefit - harm)
    R += R_Robber

    # e1. Development Cards in Hand
    w_Dev_Hand = 15
    dev_hand_value = sum([get_card_value(card) for card in state.hand.development_cards])
    R_Dev_Hand = w_Dev_Hand * dev_hand_value
    R += R_Dev_Hand

    # e2. Development Cards Played
    w_Dev_Played = 25
    dev_played_value = sum([get_played_card_value(card) for card in state.played_development_cards])
    R_Dev_Played = w_Dev_Played * dev_played_value
    R += R_Dev_Played

    # f. Hand Size Penalty
    w_HandSize = -15  # Negative weight for penalty
    hand_size = sum(state.hand.resources.values())
    penalty = max(0, hand_size - 7)
    R_HandSize = w_HandSize * penalty
    R += R_HandSize

    # g. Build Capability Rewards
    # Determine if the player can build various structures
    can_build_road = can_build(state.hand.resources, 'road')
    can_build_settlement = can_build(state.hand.resources, 'settlement')
    can_build_city = can_build(state.hand.resources, 'city')
    can_purchase_dev = can_build(state.hand.resources, 'development_card')

    # Assign rewards based on build capabilities
    R_Build = (
        5 * can_build_road +
        20 * can_build_settlement +
        25 * can_build_city +
        10 * can_purchase_dev
    )
    R += R_Build

    return R

# Helper Functions (to be implemented based on game mechanics)

def get_tile_value(tile):
    # Example: Tile value based on dice probability
    dice_probabilities = {
        2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
        8: 5, 9: 4, 10: 3, 11: 2, 12: 1
    }
    return dice_probabilities.get(tile.number, 0)

def calculate_harm(robber_tile, state):
    # Calculate the negative impact on the player
    harmed_resources = state.player_tile_resources.get(robber_tile, 0)
    return harmed_resources

def calculate_benefit(robber_tile, opponents):
    # Calculate the positive impact by hindering opponents
    benefit = 0
    for opponent in opponents:
        benefit += opponent.tile_resources.get(robber_tile, 0)
    return benefit

def get_card_value(card):
    # Assign value based on card type
    card_values = {
        'Knight': 2,
        'VictoryPoint': 5,
        'RoadBuilding': 3,
        'YearOfPlenty': 3,
        'Monopoly': 4
    }
    return card_values.get(card.type, 0)

def get_played_card_value(card):
    # Assign value based on played card type
    played_card_values = {
        'Knight': 2,  # For largest army
        'VictoryPoint': 5,
        'RoadBuilding': 3,
        'YearOfPlenty': 3,
        'Monopoly': 4
    }
    return played_card_values.get(card.type, 0)

def can_build(resources, structure_type):
    # Define the resource costs for each structure
    cost = {
        'road': {'brick': 1, 'wood': 1},
        'settlement': {'brick': 1, 'wood': 1, 'sheep': 1, 'wheat': 1},
        'city': {'wheat': 2, 'ore': 3},
        'development_card': {'sheep': 1, 'wheat': 1, 'ore': 1}
    }

    required = cost.get(structure_type, {})
    for resource, amount in required.items():
        if resources.get(resource, 0) < amount:
            return 0  # Cannot build
    return 1  # Can build

# Example State Class (for clarity; in practice, your state representation may differ)

class State:
    def __init__(self, victory_points, settlements, cities, hand, robber_position, opponents, played_development_cards):
        self.victory_points = victory_points
        self.settlements = settlements  # List of Settlement objects
        self.cities = cities            # List of City objects
        self.hand = hand                # Hand object
        self.robber_position = robber_position
        self.opponents = opponents        # List of Opponent objects
        self.played_development_cards = played_development_cards  # List of played DevelopmentCard objects

class Settlement:
    def __init__(self, tile):
        self.tile = tile

class City:
    def __init__(self, tile):
        self.tile = tile

class Hand:
    def __init__(self, resources, development_cards):
        self.resources = resources  # Dict like {'brick': 2, 'wood': 3, ...}
        self.development_cards = development_cards  # List of DevelopmentCard objects

class Opponent:
    def __init__(self, tile_resources):
        self.tile_resources = tile_resources  # Dict mapping tile to resources

class DevelopmentCard:
    def __init__(self, type):
        self.type = type  # e.g., 'Knight', 'VictoryPoint', etc.