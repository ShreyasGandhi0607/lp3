class ItemValue:
    def __init__(self, profit, weight):
        self.profit = profit  # profit value of item
        self.weight = weight  # weight of item

    # Method to calculate profit-to-weight ratio
    def cost(self):
        return self.profit / self.weight

# Function to get the maximum value that can be carried in the knapsack
def get_max_value(items, capacity):
    # Sort items by profit-to-weight ratio in descending order
    items.sort(key=lambda x: x.cost(), reverse=True)

    total_value = 0.0

    for item in items:
        if capacity - item.weight >= 0:
            # This weight can be picked completely
            capacity -= item.weight
            total_value += item.profit
        else:
            # Item can't be picked whole; take fraction of it
            fraction = capacity / item.weight
            total_value += item.profit * fraction
            break  # Knapsack is full

    return total_value

# Driver code
if __name__ == "__main__":
    items = [ItemValue(60, 10), ItemValue(100, 20), ItemValue(120, 30)]
    capacity = 50

    max_value = get_max_value(items, capacity)
    print(f"Maximum value in Knapsack = {max_value}")
