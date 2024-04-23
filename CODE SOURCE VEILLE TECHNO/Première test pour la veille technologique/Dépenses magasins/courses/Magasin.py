class Magasin:
    def __init__(self, pants_coeff, apples_coeff, restaurant_coeff, oil_coeff, beef_coeff, water_bottle_coeff,
                 lamp_coeff):
        # Initial prices and scores
        self.Pants_TeeShirtPrice = 50 * pants_coeff
        self.ApplesPrice = 12 * apples_coeff
        self.RestaurantPrice = 20 * restaurant_coeff
        self.OilPrice = 100 * oil_coeff
        self.BeefPrice = 30 * beef_coeff
        self.WaterBootlePrice = 10 * water_bottle_coeff
        self.LampPrice = 5 * lamp_coeff

        self.Pants_TeeShirtScore = 10
        self.ApplesScore = 5
        self.RestaurantScore = 15
        self.OilScore = 10
        self.BeefScore = 7
        self.WaterBootleScore = 10
        self.LampScore = 20

        self.total_score = 0
        self.financialScore = 0

    def calculate_total_score(self):
        self.total_score = 0

        self.total_score += self.Pants_TeeShirtScore * (1 / self.Pants_TeeShirtPrice)
        self.total_score += self.ApplesScore * (1 / self.ApplesPrice)
        self.total_score += self.RestaurantScore * (1 / self.RestaurantPrice)
        self.total_score += self.OilScore * (1 / self.OilPrice)
        self.total_score += self.BeefScore * (1 / self.BeefPrice)
        self.total_score += self.WaterBootleScore * (1 / self.WaterBootlePrice)
        self.total_score += self.LampScore * (1 / self.LampPrice)
