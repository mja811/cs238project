# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 17:32:30 2025

@author: Barney
"""

import numpy as np
import pandas as pd
import random
from math import radians, sin, cos, sqrt, atan2

# ----------------------------
# Load Data
# ----------------------------
df_conditions = pd.read_csv("conditions.csv")  
df_temps = pd.read_csv("temperature.csv")     
df_stadia = pd.read_csv("stadium.csv")         

stadiums = list(df_stadia['Name'])
num_stadiums = len(stadiums)

coords = {row['Name']: (row['Latitude'], row['Longitude']) for _, row in df_stadia.iterrows()}
capacities = {row['Name']: row['Capacity'] for _, row in df_stadia.iterrows()}
roof_types = {row['Name']: row.get('Roof', 'outdoor').lower() for _, row in df_stadia.iterrows()}

# Merge weather & temperature
weather = pd.merge(df_conditions, df_temps, on=['stadium_name', 'week_number'], how='inner')
weather = weather.set_index(['stadium_name', 'week_number'])
weather = weather.reindex(
    pd.MultiIndex.from_product([stadiums, range(1, 53)], names=['stadium_name', 'week_number']),
    fill_value=np.nan
).reset_index()
weather['condition'] = weather['condition'].fillna('unknown')
weather['average_temperature'] = weather['average_temperature'].fillna(10)

# ----------------------------
# Reward & Financial Components
# ----------------------------
condition_values = {
    'sunny': 1.0, 'clear': 1.0, 'partly cloudy': 0.9, 'cloudy': 0.75,
    'drizzle': 0.5, 'rainy': 0.3, 'snowy': 0.05, 'unknown': 0.5
}

fuel_cost = 2.0
fixed_cost = 50000
ticket_price = 250
merch_price = 60
b = 0.9
alpha = 0.6  # weighting factor for environmental scaling

# ----------------------------
# Helper Functions
# ----------------------------
def haversine_distance(s1, s2):
    lat1, lon1 = coords[s1]
    lat2, lon2 = coords[s2]
    R = 6371
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def weather_score(stadium, week):
    row = weather[(weather['stadium_name'] == stadium) & (weather['week_number'] == week)]
    cond = str(row['condition'].values[0]).lower()
    temp = row['average_temperature'].values[0]

    if roof_types.get(stadium, 'outdoor') == 'indoor':
        return 1.0
    condition_value = condition_values.get(cond, 0.5)
    score = (temp / 35) * condition_value
    return min(max(score, 0), 1)

# ----------------------------
# Reward Definitions
# ----------------------------
def total_reward(curr, nxt, week, num_concerts):
    """Full reward: combines ticket, merch, and cost with α scaling."""
    ws = weather_score(nxt, week)
    demand = capacities[nxt] * ws
    ticket_revenue = demand * ticket_price
    merch_revenue = demand * merch_price * (b ** num_concerts)
    if curr == nxt:
        travel_cost = fixed_cost
    else:
        dist = haversine_distance(curr, nxt)
        travel_cost = fuel_cost * dist + fixed_cost
    return alpha * (ticket_revenue + merch_revenue) - travel_cost

def financial_reward(curr, nxt, week, num_concerts):
    """Pure financial objective: maximize ticket + merch − cost (no α scaling)."""
    ws = weather_score(nxt, week)
    demand = capacities[nxt] * ws
    ticket_revenue = demand * ticket_price
    merch_revenue = demand * merch_price * (b ** num_concerts)
    if curr == nxt:
        travel_cost = fixed_cost
    else:
        dist = haversine_distance(curr, nxt)
        travel_cost = fuel_cost * dist + fixed_cost
    return (ticket_revenue + merch_revenue) - travel_cost

# ----------------------------
# Q-Learning Setup
# ----------------------------
num_weeks = 52
alpha_q = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 5000

Q_reward = np.zeros((num_weeks, num_stadiums, num_stadiums))
Q_financial = np.zeros((num_weeks, num_stadiums, num_stadiums))

def choose_action(Q, curr_idx, week):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_stadiums - 1)
    return np.argmax(Q[week, curr_idx])

# ----------------------------
# Train Both Policies
# ----------------------------
for ep in range(episodes):
    curr_idx_r = random.randint(0, num_stadiums - 1)
    curr_idx_f = random.randint(0, num_stadiums - 1)
    num_concerts_r = 0
    num_concerts_f = 0

    for week in range(num_weeks):
        # --- Reward-based ---
        act_idx_r = choose_action(Q_reward, curr_idx_r, week)
        curr_r, next_r = stadiums[curr_idx_r], stadiums[act_idx_r]
        num_concerts_r += 1
        r1 = total_reward(curr_r, next_r, week+1, num_concerts_r)
        if week < num_weeks - 1:
            Q_reward[week, curr_idx_r, act_idx_r] += alpha_q * (r1 + gamma * np.max(Q_reward[week+1, act_idx_r]) - Q_reward[week, curr_idx_r, act_idx_r])
        else:
            Q_reward[week, curr_idx_r, act_idx_r] += alpha_q * (r1 - Q_reward[week, curr_idx_r, act_idx_r])
        curr_idx_r = act_idx_r

        # --- Financial-based ---
        act_idx_f = choose_action(Q_financial, curr_idx_f, week)
        curr_f, next_f = stadiums[curr_idx_f], stadiums[act_idx_f]
        num_concerts_f += 1
        r2 = financial_reward(curr_f, next_f, week+1, num_concerts_f)
        if week < num_weeks - 1:
            Q_financial[week, curr_idx_f, act_idx_f] += alpha_q * (r2 + gamma * np.max(Q_financial[week+1, act_idx_f]) - Q_financial[week, curr_idx_f, act_idx_f])
        else:
            Q_financial[week, curr_idx_f, act_idx_f] += alpha_q * (r2 - Q_financial[week, curr_idx_f, act_idx_f])
        curr_idx_f = act_idx_f

# ----------------------------
# Derive Policies and Save
# ----------------------------
def extract_policy(Q, reward_func, label):
    curr_idx = 0
    best_tour = []
    cumulative_rewards = []
    cum_reward = 0
    num_concerts = 0

    for week in range(num_weeks):
        next_idx = np.argmax(Q[week, curr_idx])
        best_tour.append(stadiums[next_idx])
        num_concerts += 1
        r = reward_func(stadiums[curr_idx], stadiums[next_idx], week+1, num_concerts)
        cum_reward += r
        cumulative_rewards.append({
            'Week': week+1,
            'Stadium': stadiums[next_idx],
            'Reward': r,
            'CumulativeReward': cum_reward
        })
        curr_idx = next_idx

    df = pd.DataFrame(cumulative_rewards)
    df.to_csv(f"{label}.csv", index=False)
    print(f"✅ Saved {label}.csv")
    return best_tour, cum_reward

best_reward_tour, reward_total = extract_policy(Q_reward, total_reward, "policy_reward_based")
best_financial_tour, financial_total = extract_policy(Q_financial, financial_reward, "policy_financial_based")

# ----------------------------
# Display Results
# ----------------------------
print("\nOptimal Reward-based Tour:")
for i, s in enumerate(best_reward_tour, start=1):
    print(f"Week {i}: {s}")
print(f"\nTotal Reward-based Score: {reward_total:,.2f}")

print("\nOptimal Financial-based Tour:")
for i, s in enumerate(best_financial_tour, start=1):
    print(f"Week {i}: {s}")
print(f"\nTotal Financial Profit: {financial_total:,.2f}")
