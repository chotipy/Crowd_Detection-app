def classify_crowd(people_count: int) -> str:
    if people_count <= 3:
        return "Sedikit"
    elif people_count <= 30:
        return "Sedang"
    else:
        return "Ramai"
