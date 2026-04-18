---
name: wargame_query
description: Tactical map query skill for accessing war game state information. Use when needing information about unit positions, friendly forces, hostile forces, or tactical situation from the interactive map.
---

# Tactical Map Query Skill

## Overview

This skill provides access to the tactical map state, showing military unit positions with NATO symbols. Use it to get information about friendly and hostile forces for comprehensive battlefield awareness.

## Context Awareness

The system context header indicates if the tactical map is available. If available, you'll see unit counts. If not available, focus on video and PDF analysis instead.

## Available Actions

### 1. tactical_situation
Get overall tactical overview.

```python
result = wargame_query_skill("tactical_situation")
# Returns: map_center, total_units, friendly_count, hostile_count, summary
```

### 2. friendly_units
Get all friendly (blue force) units. Use for "our forces" queries.

```python
result = wargame_query_skill("friendly_units")
# Returns: list of friendly units with names, types, locations
```

**Note:** Friendly unit names typically start with "F-" (e.g., F-1, F-2)

### 3. hostile_units
Get all hostile (red force/enemy) units.

```python
result = wargame_query_skill("hostile_units")
# Returns: list of enemy units with names, types, locations
```

**Note:** Hostile unit names typically start with "H-" (e.g., H-1, H-2)

### 4. unit_details
Get detailed information about a specific unit.

```python
result = wargame_query_skill("unit_details", unit_name="Alpha Company")
# Returns: id, name, affiliation, unit_type, echelon, sidc, location, waypoints
```

### 5. unit_waypoints
Get movement waypoints (planned route) for a unit.

```python
result = wargame_query_skill("unit_waypoints", unit_name="Bravo Platoon")
# Returns: unit_location, waypoints list, num_waypoints
```

### 6. units_by_type
Find all units of a specific type.

```python
result = wargame_query_skill("units_by_type", unit_type="ARMOR")
# Returns: units with friendly_count, hostile_count breakdowns
```

## Valid Unit Types

- INFANTRY, ARMOR, ARTILLERY, AIR_DEFENSE
- AVIATION, ENGINEER, RECONNAISSANCE, SIGNAL
- MEDICAL, SUPPLY, HEADQUARTERS, MECHANIZED
- MOTORIZED, AIRBORNE, SPECIAL_FORCES, NAVY
- CYBER, ELECTRONIC_WARFARE, UAV

## Critical: Keep Results Separate

When comparing friendly and hostile forces, **never mix up the results**:

```python
# Get both sides - KEEP SEPARATE!
friendly_result = wargame_query_skill("friendly_units")
hostile_result = wargame_query_skill("hostile_units")

# Process friendly units from friendly_result ONLY
friendly_units = friendly_result["result"]["units"]

# Process hostile units from hostile_result ONLY
hostile_units = hostile_result["result"]["units"]

# Report separately
print("FRIENDLY FORCES:")
for unit in friendly_units:
    print(f"  {unit['name']} at {unit['location']}")

print("HOSTILE FORCES:")
for unit in hostile_units:
    print(f"  {unit['name']} at {unit['location']}")
```

## Integrating with Video Analysis

Correlate video findings with map positions:

```python
# Step 1: Find tanks in video
video_tanks = videodb_query_skill("object_search", object_type="tank")

# Step 2: Get armor units on tactical map
map_armor = wargame_query_skill("units_by_type", unit_type="ARMOR")

# Step 3: Correlate and report
```

## Error Handling

```python
result = wargame_query_skill("tactical_situation")

if result["status"] == "error":
    # Map may not be initialized
    print("Tactical map not available")
    # Proceed with video-only analysis
elif result["status"] == "success":
    if result["result"]["total_units"] == 0:
        print("No units on tactical map")
    else:
        # Process tactical data
```

## Best Practices

1. Check if tactical map is available before querying
2. Keep friendly and hostile results separate
3. Include unit movement info (waypoints) when relevant
4. Correlate map data with video analysis
5. Always specify which side (friendly/hostile) units belong to
