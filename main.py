
import os
import csv
import io
import urllib.request
import urllib.error
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(
    title="North India Air Intelligence API",
    description="NASA FIRMS satellite fire monitoring across North India",
    version="1.0.0"
)

NASA_KEY = os.getenv("NASA_FIRMS_MAP_KEY")
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

REGIONS = {
    "Indian Punjab": [73.8, 29.5, 77.0, 32.6],
    "Haryana": [74.4, 27.6, 77.6, 30.9],
    "Rajasthan": [69.3, 23.0, 78.3, 30.2],
    "Madhya Pradesh": [74.0, 21.0, 82.0, 26.9],
    "Pakistan Punjab": [69.0, 27.5, 75.5, 34.0],
    "Delhi NCR": [76.5, 27.0, 78.0, 29.5],
    "Western Uttar Pradesh": [77.0, 25.5, 81.5, 30.5],
}

SOURCES = {
    "NOAA20": "VIIRS_NOAA20_NRT",
    "NOAA21": "VIIRS_NOAA21_NRT",
    "SNPP": "VIIRS_SNPP_NRT",
}


@app.get("/")
def root():
    return {
        "message": "North India Air Intelligence API",
        "status": "operational",
        "live_data": "Use /api/fires",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "nasa_key_configured": bool(NASA_KEY),
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/regions")
def get_regions():
    return {
        "regions": [
            {
                "name": name,
                "bounds": bounds
            }
            for name, bounds in REGIONS.items()
        ]
    }


def get_region(latitude, longitude):
    matches = []

    for name, bounds in REGIONS.items():
        west, south, east, north = bounds

        if south <= latitude <= north and west <= longitude <= east:
            matches.append(name)

    return matches



@app.get("/api/fires")
def get_fires(
    days: int = Query(default=1, ge=1, le=5),
    satellite: str = Query(default="NOAA21"),
    min_confidence: str = Query(default="low")
):
    if not NASA_KEY:
        raise HTTPException(
            status_code=503,
            detail="NASA FIRMS key is not configured on the server."
        )

    if satellite not in SOURCES:
        raise HTTPException(
            status_code=400,
            detail="Choose NOAA20, NOAA21 or SNPP."
        )

    if min_confidence not in ("low", "nominal", "high"):
        raise HTTPException(
            status_code=400,
            detail="Confidence must be low, nominal or high."
        )

    # Combined North India monitoring bounding box
    area = "68,20,83,35"
    source = SOURCES[satellite]

    url = (
        f"{FIRMS_BASE}/{NASA_KEY}/"
        f"{source}/{area}/{days}"
    )

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "NorthIndiaAirIntelligence/1.0"}
        )

        with urllib.request.urlopen(request, timeout=45) as response:
            content = response.read().decode("utf-8")

        reader = csv.DictReader(io.StringIO(content))

        required_columns = {
            "latitude", "longitude", "confidence",
            "acq_date", "acq_time", "frp"
        }

        if not reader.fieldnames or not required_columns.issubset(
            set(reader.fieldnames)
        ):
            raise HTTPException(
                status_code=502,
                detail="NASA FIRMS returned an unexpected data format."
            )

        detections = []

        confidence_map = {
            "l": "low",
            "n": "nominal",
            "h": "high",
            "low": "low",
            "nominal": "nominal",
            "high": "high"
        }

        for row in reader:
            try:
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
            except (ValueError, KeyError, TypeError):
                continue

            regions = get_region(latitude, longitude)

            if not regions:
                continue

            confidence_code = str(
                row.get("confidence", "unknown")
            ).strip().lower()

            confidence = confidence_map.get(
                confidence_code, "unknown"
            )

            if min_confidence == "nominal" and confidence not in (
                "nominal", "high"
            ):
                continue

            if min_confidence == "high" and confidence != "high":
                continue

            try:
                frp = float(row.get("frp") or 0)
            except (ValueError, TypeError):
                frp = 0

            detections.append({
                "latitude": latitude,
                "longitude": longitude,
                "regions": regions,
                "acquisition_date": row.get("acq_date"),
                "acquisition_time_utc": row.get("acq_time"),
                "satellite": row.get("satellite"),
                "confidence": confidence,
                "frp_mw": frp,
                "daynight": row.get("daynight"),
                "instrument": source
            })

        return {
            "status": "live",
            "source": "NASA FIRMS",
            "satellite": satellite,
            "period_days": days,
            "count": len(detections),
            "retrieved_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "regional_assignment": "Approximate rectangular boundaries; regional counts may overlap",
            "detections": detections
        }

    except urllib.error.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"NASA FIRMS returned HTTP {error.code}."
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="NASA FIRMS request failed. Check server logs."
        )

@app.get("/api/fires/summary")
def fire_summary(
    days: int = Query(default=1, ge=1, le=5),
    satellite: str = Query(default="NOAA21")
):
    result = get_fires(
        days=days,
        satellite=satellite,
        min_confidence="low"
    )

    counts = {name: 0 for name in REGIONS}

    for detection in result["detections"]:
        for region in detection["regions"]:
            counts[region] += 1

    return {
        "status": "live",
        "source": "NASA FIRMS",
        "satellite": satellite,
        "period_days": days,
        "total_detections": result["count"],
        "regional_counts": counts,
        "retrieved_at_utc": result["retrieved_at_utc"]
    }
