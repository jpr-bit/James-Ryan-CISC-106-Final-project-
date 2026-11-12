from dataclasses import dataclass
import matplotlib.pyplot as plt
from drafter import *
from bakery import assert_equal


# ==========================================
# 1. DATACLASSES
# ==========================================

@dataclass
class PredictionRecord:
    workout_label: str
    target_distance: int
    predicted_split: str
    total_time: str
    watts: float
    hr_note: str
    is_goal_met: bool


@dataclass
class State:
    piece_unit: str       # "Meters" or "Minutes"
    reps: int             # 1 for single piece, 2+ for intervals
    rep_amount: float     # Distance in meters OR duration in minutes
    rest_input: str       # Stores user's raw rest text (e.g., "1:30", "2", "90s", or "0")
    split_minutes: int
    split_seconds: float
    target_distance: int
    heart_rate: int       # 0 if left blank
    weight_lbs: float
    saved_predictions: list[PredictionRecord]


# ==========================================
# 2. SIMPLE MATH & SMART REST PARSER
# ==========================================

def to_float(text: str, fallback: float) -> float:
    """Safely converts text to a float, returning fallback if invalid."""
    cleaned = text.strip()
    if cleaned and cleaned.replace(".", "", 1).isdigit():
        return float(cleaned)
    return fallback


def to_int(text: str, fallback: int) -> int:
    """Safely converts text to an integer, returning fallback if invalid."""
    cleaned = text.strip()
    if cleaned.isdigit():
        return int(cleaned)
    return fallback


def parse_rest_minutes(text: str) -> float:
    """
    Accepts ANY natural rest format and converts it into minutes:
    - "1:30" -> 1.5 minutes (MM:SS clock format)
    - "90s"  -> 1.5 minutes (seconds ending in 's')
    - "2" or "1.5" -> 2.0 or 1.5 minutes (plain minutes)
    - "" or "0" -> 0.0 minutes
    """
    cleaned = text.strip().lower()
    if not cleaned:
        return 0.0

    # Case 1: Clock format like "1:30" or "0:45"
    if ":" in cleaned:
        parts = cleaned.split(":")
        if len(parts) == 2:
            mins = to_int(parts[0], 0)
            secs = to_float(parts[1], 0.0)
            return mins + (secs / 60.0)
        return 0.0

    # Case 2: Seconds shorthand like "90s" or "45s"
    if cleaned.endswith("s"):
        secs = to_float(cleaned[:-1], 0.0)
        return secs / 60.0

    # Case 3: Plain or decimal minutes like "2" or "1.5"
    return to_float(cleaned, 0.0)


def format_rest_label(rest_mins: float) -> str:
    """Formats rest minutes cleanly as M:SS (e.g., 1.5 -> '1:30', 2.0 -> '2:00')."""
    total_sec = int(round(rest_mins * 60))
    mins = total_sec // 60
    secs = total_sec % 60
    if secs < 10:
        return f"{mins}:0{secs}"
    return f"{mins}:{secs}"


def calculate_watts(split_sec: float) -> float:
    """Concept2 formula: Watts = 2.80 / (split / 500)^3."""
    if split_sec <= 0:
        return 0.0
    pace = split_sec / 500.0
    return round(2.80 / (pace ** 3), 1)


def get_rep_meters(unit: str, amount: float, split_sec: float) -> float:
    """Returns meters rowed in one rep (converts minutes to meters if timed)."""
    if unit == "Minutes" and split_sec > 0:
        duration_sec = amount * 60.0
        return (duration_sec / split_sec) * 500.0
    return amount


def predict_split(
    split_sec: float,
    unit: str,
    reps: int,
    rep_amount: float,
    rest_mins: float,
    target_dist: int,
    bpm: int,
) -> float:
    """Predicts target 500m split from any distance or time-based workout."""
    rep_meters = get_rep_meters(unit, rep_amount, split_sec)
    total_meters = reps * rep_meters
    if total_meters <= 0 or target_dist <= 0:
        return split_sec

    # 1. Interval Rest Adjustment
    rest_penalty = 0.0
    if reps > 1 and rest_mins > 0:
        rest_penalty = min(8.0, (reps - 1) * 0.30 * rest_mins)

    equiv_continuous_split = split_sec + rest_penalty

    # 2. Riegel's Distance Scaling
    predicted = equiv_continuous_split * ((target_dist / total_meters) ** 0.06)

    # 3. Optional Submax Heart Rate Boost
    if 0 < bpm < 160 and predicted > 5.0:
        predicted = predicted - 0.5

    return predicted


def make_workout_label(unit: str, reps: int, amount: float, rest_mins: float) -> str:
    """Creates a clean label like '2000m', '8x500m (1:30 r)', or '3x10min (2:00 r)'."""
    amt_str = str(int(amount)) if amount == int(amount) else str(amount)
    suffix = "min" if unit == "Minutes" else "m"

    if reps <= 1 or rest_mins <= 0:
        return f"{amt_str}{suffix}" if reps <= 1 else f"{reps}x{amt_str}{suffix}"
    return f"{reps}x{amt_str}{suffix} ({format_rest_label(rest_mins)} r)"


def format_hr(bpm: int, watts: float) -> str:
    """Returns a short heart rate summary string."""
    if bpm <= 0:
        return "None"
    wpb = round(watts / bpm, 2)
    if bpm < 160:
        return f"{bpm} BPM (Submax | {wpb} W/BPM)"
    elif bpm < 180:
        return f"{bpm} BPM (Threshold | {wpb} W/BPM)"
    return f"{bpm} BPM (Max Effort | {wpb} W/BPM)"


def format_time(total_seconds: float) -> str:
    """Formats seconds into PM5 monitor format (M:SS.d)."""
    rounded = round(total_seconds, 1)
    mins = int(rounded // 60)
    secs = round(rounded % 60, 1)
    if secs < 10:
        return f"{mins}:0{secs}"
    return f"{mins}:{secs}"


# ==========================================
# 3. DRAFTER ROUTES (8 ROUTES / 6 PAGES)
# ==========================================

@route
def index(state: State) -> Page:
    """Page 1: Streamlined input form with effortless rest entry."""
    hr_text = str(state.heart_rate) if state.heart_rate > 0 else ""

    return Page(
        state,
        [
            Header("Concept2 Erg Score Predictor"),
            "1. Workout Type (Distance in Meters OR Duration in Minutes):",
            SelectBox("unit", ["Meters", "Minutes"], state.piece_unit),
            "2. Number of Intervals (enter 1 for a single piece):",
            TextBox("reps_str", str(state.reps)),
            "3. Distance (m) or Time (min) per Interval (e.g., 500, 2000, or 20):",
            TextBox("amount_str", str(state.rep_amount)),
            "4. Rest Time — type 1:30, 2 (mins), 90s, or leave blank for none:",
            TextBox("rest_str", state.rest_input),
            "5. Average 500m Split — Minutes & Seconds:",
            TextBox("mins_str", str(state.split_minutes)),
            TextBox("secs_str", str(state.split_seconds)),
            "6. Target Distance to Predict in Meters (e.g., 2000 or 6000):",
            TextBox("target_str", str(state.target_distance)),
            "7. Average Heart Rate in BPM (Optional — leave blank if none):",
            TextBox("hr_str", hr_text),
            Button("Calculate Prediction", "calculate_score"),
            Button("View Score Graph", "view_graph"),
            Button("View Saved History", "view_history"),
        ],
    )


@route
def calculate_score(
    state: State,
    unit: str,
    reps_str: str,
    amount_str: str,
    rest_str: str,
    mins_str: str,
    secs_str: str,
    target_str: str,
    hr_str: str,
) -> Page:
    """Page 2: Updates state, computes prediction, and saves to history."""
    state.piece_unit = unit
    state.reps = max(1, to_int(reps_str, state.reps))
    state.rep_amount = max(1.0, to_float(amount_str, state.rep_amount))
    state.rest_input = "" if state.reps == 1 else rest_str.strip()
    state.split_minutes = to_int(mins_str, state.split_minutes)
    state.split_seconds = to_float(secs_str, state.split_seconds)
    state.target_distance = max(100, to_int(target_str, state.target_distance))
    state.heart_rate = to_int(hr_str, 0)

    rest_mins = parse_rest_minutes(state.rest_input)
    known_split_sec = (state.split_minutes * 60) + state.split_seconds
    pred_split_sec = predict_split(
        known_split_sec,
        state.piece_unit,
        state.reps,
        state.rep_amount,
        rest_mins,
        state.target_distance,
        state.heart_rate,
    )
    total_sec = pred_split_sec * (state.target_distance / 500.0)

    split_text = format_time(pred_split_sec)
    total_text = format_time(total_sec)
    pred_watts = calculate_watts(pred_split_sec)
    known_watts = calculate_watts(known_split_sec)
    hr_note = format_hr(state.heart_rate, known_watts)
    label = make_workout_label(
        state.piece_unit, state.reps, state.rep_amount, rest_mins
    )

    state.saved_predictions.append(
        PredictionRecord(
            workout_label=label,
            target_distance=state.target_distance,
            predicted_split=split_text,
            total_time=total_text,
            watts=pred_watts,
            hr_note=hr_note,
            is_goal_met=False,
        )
    )

    return Page(
        state,
        [
            Header("Prediction Results"),
            f"Workout Evaluated: {label} -> Target: {state.target_distance}m",
            f"Predicted Split: {split_text} /500m",
            f"Predicted Finish Time: {total_text}",
            f"Target Power: {pred_watts} Watts",
            f"Heart Rate Info: {hr_note}",
            Button("Weight-Adjust Score", "weight_adjust_page"),
            Button("View Score Graph", "view_graph"),
            Button("View Saved History", "view_history"),
            Button("Back to Home", "index"),
        ],
    )


@route
def view_graph(state: State) -> Page:
    """Page 3: Plots a Matplotlib bar chart of all evaluated scores."""
    if not state.saved_predictions:
        return Page(
            state,
            [
                Header("Evaluated Scores Graph"),
                "No predictions yet! Calculate a workout first.",
                Button("Back to Home", "index"),
            ],
        )

    labels: list[str] = []
    watts_list: list[float] = []
    colors: list[str] = []

    i = 0
    for rec in state.saved_predictions:
        labels.append(f"#{i}: {rec.target_distance}m ({rec.predicted_split})\n[{rec.workout_label}]")
        watts_list.append(rec.watts)
        colors.append("#2ca02c" if rec.is_goal_met else "#1f77b4")
        i += 1

    plt.clf()
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, watts_list, color=colors)
    plt.ylabel("Predicted Power (Watts)")
    plt.title("Evaluated Erg Predictions (Blue = Active, Green = Goal Met)")

    for bar in bars:
        w = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, w + 3, f"{w}W", ha="center")

    plt.ylim(0, max(watts_list) * 1.18)
    plt.tight_layout()

    return Page(
        state,
        [
            Header("Evaluated Scores Graph"),
            MatPlotLibPlot(),
            Button("View Saved History", "view_history"),
            Button("Back to Home", "index"),
        ],
    )


@route
def weight_adjust_page(state: State) -> Page:
    """Page 4: Input form for body weight and dynamic slides."""
    return Page(
        state,
        [
            Header("Weight & Slide Adjustment"),
            "Body Weight (lbs):",
            TextBox("weight_str", str(state.weight_lbs)),
            "Rowed on Dynamic / Slides?",
            CheckBox("use_slides", False),
            Button("Calculate Adjusted Score", "calculate_weight_adjust"),
            Button("Back to Home", "index"),
        ],
    )


@route
def calculate_weight_adjust(state: State, weight_str: str, use_slides: bool) -> Page:
    """Page 5: Calculates Concept2 weight-adjusted split and W/kg."""
    state.weight_lbs = max(50.0, to_float(weight_str, state.weight_lbs))

    rest_mins = parse_rest_minutes(state.rest_input)
    known_split_sec = (state.split_minutes * 60) + state.split_seconds
    raw_split = predict_split(
        known_split_sec,
        state.piece_unit,
        state.reps,
        state.rep_amount,
        rest_mins,
        state.target_distance,
        state.heart_rate,
    )

    if use_slides and raw_split > 2.0:
        raw_split -= 1.5

    wf = (state.weight_lbs / 270.0) ** 0.222
    adj_split = raw_split * wf
    adj_total = adj_split * (state.target_distance / 500.0)
    wpk = round(calculate_watts(raw_split) / (state.weight_lbs / 2.20462), 2)

    return Page(
        state,
        [
            Header("Weight-Adjusted Score"),
            f"Weight Factor (Wf): {round(wf, 3)} (at {state.weight_lbs} lbs)",
            f"Weight-Adjusted Split: {format_time(adj_split)} /500m",
            f"Weight-Adjusted Total: {format_time(adj_total)}",
            f"Power-to-Weight: {wpk} W/kg",
            Button("Back to Home", "index"),
        ],
    )


@route
def view_history(state: State) -> Page:
    """Page 6: Loops through saved PredictionRecord list and lets user mark goals met."""
    if not state.saved_predictions:
        return Page(
            state,
            [
                Header("Saved Predictions Log"),
                "No predictions saved yet!",
                Button("Back to Home", "index"),
            ],
        )

    items: list[str] = []
    i = 0
    for rec in state.saved_predictions:
        status = "Goal Met!" if rec.is_goal_met else "In Progress"
        items.append(
            f"#{i}: {rec.workout_label} -> {rec.target_distance}m | "
            f"{rec.predicted_split}/500m ({rec.total_time}) | "
            f"{rec.watts}W | HR: {rec.hr_note} | {status}"
        )
        i += 1

    return Page(
        state,
        [
            Header("Saved Predictions Log"),
            BulletedList(items),
            "Prediction # to mark as Goal Met:",
            TextBox("goal_idx", "0"),
            Button("Mark Goal Met", "mark_goal_met"),
            Button("View Score Graph", "view_graph"),
            Button("Clear History", "clear_history"),
            Button("Back to Home", "index"),
        ],
    )


@route
def mark_goal_met(state: State, goal_idx: str) -> Page:
    """Route 7: Marks a chosen PredictionRecord as achieved."""
    idx = to_int(goal_idx, -1)
    if 0 <= idx < len(state.saved_predictions):
        state.saved_predictions[idx].is_goal_met = True
    return view_history(state)


@route
def clear_history(state: State) -> Page:
    """Route 8: Clears all saved predictions."""
    state.saved_predictions = []
    return view_history(state)


# ==========================================
# 4. UNIT TESTS (RUN BEFORE start_server)
# ==========================================

# Smart Rest Parser Tests: all 4 ways of entering 1 minute 30 seconds (or 0) work identically
assert_equal(parse_rest_minutes("1:30"), 1.5)
assert_equal(parse_rest_minutes("1.5"), 1.5)
assert_equal(parse_rest_minutes("90s"), 1.5)
assert_equal(parse_rest_minutes("2"), 2.0)
assert_equal(parse_rest_minutes(""), 0.0)

assert_equal(make_workout_label("Meters", 8, 500.0, 1.5), "8x500m (1:30 r)")
assert_equal(make_workout_label("Minutes", 3, 10.0, 2.0), "3x10min (2:00 r)")

# Route test: Entering rest directly in PM5 clock style ("2:00")
assert_equal(
    calculate_score(
        State("Meters", 1, 2000.0, "", 1, 40.0, 2000, 0, 137.0, []),
        "Minutes",
        "3",
        "10",
        "2:00",
        "1",
        "45.0",
        "2000",
        "",
    ),
    Page(
        State(
            "Minutes",
            3,
            10.0,
            "2:00",
            1,
            45.0,
            2000,
            0,
            137.0,
            [
                PredictionRecord(
                    "3x10min (2:00 r)",
                    2000,
                    "1:37.9",
                    "6:31.5",
                    373.2,
                    "None",
                    False,
                )
            ],
        ),
        [
            Header("Prediction Results"),
            "Workout Evaluated: 3x10min (2:00 r) -> Target: 2000m",
            "Predicted Split: 1:37.9 /500m",
            "Predicted Finish Time: 6:31.5",
            "Target Power: 373.2 Watts",
            "Heart Rate Info: None",
            Button("Weight-Adjust Score", "weight_adjust_page"),
            Button("View Score Graph", "view_graph"),
            Button("View Saved History", "view_history"),
            Button("Back to Home", "index"),
        ],
    ),
)


# ==========================================
# 5. START SERVER
# ==========================================

start_server(State("Meters", 1, 2000.0, "", 1, 40.0, 2000, 0, 137.0, []))