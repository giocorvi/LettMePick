import random
num_tries = 100000
odds = 18/37
num_wins = 0
for _ in range(num_tries):
    cash = 10
    while 0 < cash < 20:
        if random.random() < odds:
            cash += 1
        else:
            cash -= 1

    if cash == 20:
        num_wins += 1

print(f"The win rate was {(100*num_wins/num_tries):.2f}%. The original odds were {(odds*100):.2f}%")
