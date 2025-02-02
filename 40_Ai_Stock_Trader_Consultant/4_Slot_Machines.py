import random


def spin_slot_machine():
    # Define the slot machine symbols (e.g., fruits, numbers, etc.)
    symbols = ["🍒", "🍋", "🍊", "🔔", "⭐", "💎"]

    # Spin three reels
    reel1 = random.choice(symbols)
    reel2 = random.choice(symbols)
    reel3 = random.choice(symbols)

    return [reel1, reel2, reel3]


def check_outcome(reels):
    print(" | ".join(reels))
    if reels[0] == reels[1] == reels[2]:  # Winning condition (all symbols match)
        return True
    return False


def main():
    print("🎰 Welcome to the Slot Machine 🎰")
    balance = 100  # Initial balance
    bet_amount = 10  # Fixed bet amount per spin

    while balance >= bet_amount:
        print(f"\nCurrent Balance: ${balance}")
        input("Press Enter to spin the slot machine...")

        # Spin the slot machine
        reels = spin_slot_machine()
        is_winner = check_outcome(reels)

        if is_winner:
            print("🎉 You Win! 🎉")
            balance += bet_amount * 5  # Reward (e.g., 5x the bet amount)
        else:
            print("❌ You Lose!")
            balance -= bet_amount

        if balance < bet_amount:
            print("\nGame Over! You ran out of balance.")


if __name__ == "__main__":
    main()
