import typer

def validate_region(region_str):
    """Validate and parse a single region string."""
    if region_str is None:
        return None
    
    try:
        chrom, coords = region_str.split(":")
        start, end = map(int, coords.split("-"))
        if start >= end:
            raise ValueError("start must be less than end")
        return f"{chrom}:{start}-{end}"
    except ValueError as e:
        raise typer.BadParameter(
            f"Invalid region format: '{region_str}'. Expected 'chrom:start-end' (start < end)"
        )