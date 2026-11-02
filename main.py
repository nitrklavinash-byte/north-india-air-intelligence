
import os
import csv
import io
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from threading import Lock

from fastapi import FastAPI, HTTPException, Query
from shapely.geometry import shape, Point
from shapely.ops import unary_union


app = FastAPI(
    title="North India Air Intelligence API",
    description="NASA FIRMS fire monitoring with geographic regional classification",
    version="2.0.0"
)

NASA_KEY = os.getenv("NASA_FIRMS_MAP_KEY")

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

GEOBOUNDARIES = {
    "india_states": (
        "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/"
    ),
    "pakistan_provinces": (
        "https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM1/"
    ),
    "india_districts": (
        "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/"
    )
}

SOURCES = {
    "NOAA20": "VIIRS_NOAA20_NRT",
    "NOAA21": "VIIRS_NOAA21_NRT",
    "SNPP": "VIIRS_SNPP_NRT"
}

# Western UP is a custom monitoring definition.
# It is not an official administrative region.
WEST_UP_DISTRICTS = {
    "Saharanpur",
    "Muzaffarnagar",
    "Shamli",
    "Meerut",
    "Baghpat",
    "Ghaziabad",
    "Gautam Buddha Nagar",
    "Hapur",
    "Bulandshahr",
    "Aligarh",
    "Hathras",
    "Mathura",
    "Agra",
    "Firozabad",
    "Etah",
    "Kasganj",
    "Moradabad",
    "Amroha",
    "Bijnor",
    "Rampur",
    "Sambhal",
    "Bareilly",
    "Pilibhit",
    "Shahjahanpur",
    "Budaun"
}

STATE_REGION_NAMES = {
    "Punjab": "Indian Punjab",
    "Haryana": "Haryana",
    "Rajasthan": "Rajasthan",
    "Madhya Pradesh": "Madhya Pradesh",
    "NCT of Delhi": "Delhi NCR",
    "Delhi": "Delhi NCR"
}

BOUNDARY_CACHE = None
BOUNDARY_LOCK = Lock()


def download_json(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NorthIndiaAirIntelligence/2.0"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_boundary_geojson(metadata_url):
    metadata = download_json(metadata_url)

    geojson_url = metadata.get("simplifiedGeometryGeoJSON")

    if not geojson_url:
        geojson_url = metadata.get("gjDownloadURL")

    if not geojson_url:
        raise RuntimeError("Boundary download URL is missing.")

    geojson = download_json(geojson_url)

    if "features" not in geojson:
        raise RuntimeError("Invalid boundary GeoJSON.")

    return metadata, geojson


def feature_name(properties):
    for key in (
        "shapeName",
        "NAME_1",
        "NAME_2",
        "name",
        "NAME"
    ):
        value = properties.get(key)

        if value:
            return str(value).strip()

    return ""


def load_region_geometries():
    global BOUNDARY_CACHE

    if BOUNDARY_CACHE is not None:
        return BOUNDARY_CACHE

    with BOUNDARY_LOCK:
        if BOUNDARY_CACHE is not None:
            return BOUNDARY_CACHE

        try:
            india_meta, india_data = download_boundary_geojson(
                GEOBOUNDARIES["india_states"]
            )

            pakistan_meta, pakistan_data = download_boundary_geojson(
                GEOBOUNDARIES["pakistan_provinces"]
            )

            districts_meta, districts_data = download_boundary_geojson(
                GEOBOUNDARIES["india_districts"]
            )

            regions = {}

            # Indian state and union-territory boundaries
            for feature in india_data["features"]:
                properties = feature.get("properties", {})
                state_name = feature_name(properties)

                region_name = STATE_REGION_NAMES.get(state_name)

                if region_name:
                    regions[region_name] = shape(
                        feature["geometry"]
                    )

            # Pakistan provincial boundary
            for feature in pakistan_data["features"]:
                properties = feature.get("properties", {})
                province_name = feature_name(properties)

                if province_name.lower() == "punjab":
                    regions["Pakistan Punjab"] = shape(
                        feature["geometry"]
                    )

            # Western UP: selected district polygons
            western_district_geometries = []

            for feature in districts_data["features"]:
                properties = feature.get("properties", {})
                district_name = feature_name(properties)

                parent_name = str(
                    properties.get("shapeGroup", "")
                ).strip().lower()

                if (
                    district_name in WEST_UP_DISTRICTS
                    and (
                        parent_name in ("uttar pradesh", "up")
                        or not parent_name
                    )
                ):
                    western_district_geometries.append(
                        shape(feature["geometry"])
                    )

            if not western_district_geometries:
                raise RuntimeError(
                    "Could not identify Western UP districts. "
                    "Check the district boundary properties."
                )

            regions["Western Uttar Pradesh"] = unary_union(
                western_district_geometries
            )

            required_regions = {
                "Indian Punjab",
                "Haryana",
                "Rajasthan",
                "Madhya Pradesh",
                "Delhi NCR",
                "Pakistan Punjab",
                "Western Uttar Pradesh"
            }

            missing = required_regions - set(regions.keys())

            if missing:
                raise RuntimeError(
                    "Missing boundary regions: "
                    + ", ".join(sorted(missing))
                )

            BOUNDARY_CACHE = {
                "regions": regions,
                "metadata": {
                    "india_states_year": india_meta.get(
                        "boundaryYearRepresented"
                    ),
                    "pakistan_provinces_year": pakistan_meta.get(
                        "boundaryYearRepresented"
                    ),
                    "india_districts_year": districts_meta.get(
                        "boundaryYearRepresented"
                    ),
                    "source": "geoBoundaries"
                }
            }

            return BOUNDARY_CACHE

        except Exception as error:
            raise RuntimeError(
                f"Unable to load geographic boundaries: {error}"
            )


def assign_region(latitude, longitude, region_geometries):
    point = Point(longitude, latitude)

    # Each detection is assigned to only one region.
    # This priority resolves rare shared-boundary cases.
    priority = [
        "Delhi NCR",
        "Indian Punjab",
        "Haryana",
        "Rajasthan",
        "Madhya Pradesh",
        "Western Uttar Pradesh",
        "Pakistan Punjab"
    ]

    for region_name in priority:
        geometry = region_geometries.get(region_name)

        if geometry is not None and geometry.covers(point):
            return region_name

    return None


@app.get("/")
def root():
    return {
        "message": "North India Air Intelligence API",
        "status": "operational",
        "live_data": "/api/fires",
        "documentation": "/docs",
        "version": "2.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "nasa_key_configured": bool(NASA_KEY),
        "boundary_cache_loaded": BOUNDARY_CACHE is not None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/regions")
def get_regions():
    try:
        boundary_data = load_region_geometries()

        return {
            "status": "success",
            "source": "geoBoundaries",
            "boundary_years": boundary_data["metadata"],
            "regions": [
                {
                    "name": name,
                    "type": (
                        "Custom district union"
                        if name == "Western Uttar Pradesh"
                        else "Administrative boundary"
                    )
                }
                for name in boundary_data["regions"]
            ],
            "western_up_districts": sorted(
                WEST_UP_DISTRICTS
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )


@app.get("/api/fires")
def get_fires(
    days: int = Query(default=1, ge=1, le=5),
    satellite: str = Query(default="NOAA21"),
    min_confidence: str = Query(default="low")
):
    if not NASA_KEY:
        raise HTTPException(
            status_code=503,
            detail="NASA FIRMS key is not configured."
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

    try:
        boundary_data = load_region_geometries()
        region_geometries = boundary_data["regions"]

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Geographic boundaries unavailable: {error}"
        )

    area = "68,20,83,35"
    source = SOURCES[satellite]

    url = (
        f"{FIRMS_BASE}/{NASA_KEY}/"
        f"{source}/{area}/{days}"
    )

    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "NorthIndiaAirIntelligence/2.0"
            }
        )

        with urllib.request.urlopen(request, timeout=45) as response:
            content = response.read().decode("utf-8")

        reader = csv.DictReader(io.StringIO(content))

        required_columns = {
            "latitude",
            "longitude",
            "confidence",
            "acq_date",
            "acq_time",
            "frp"
        }

        if not reader.fieldnames or not required_columns.issubset(
            set(reader.fieldnames)
        ):
            raise HTTPException(
                status_code=502,
                detail="Unexpected NASA FIRMS data format."
            )

        confidence_map = {
            "l": "low",
            "n": "nominal",
            "h": "high",
            "low": "low",
            "nominal": "nominal",
            "high": "high"
        }

        detections = []
        valid_raw_count = 0
        unassigned_count = 0

        for row in reader:
            try:
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
            except (ValueError, KeyError, TypeError):
                continue

            valid_raw_count += 1

            confidence_code = str(
                row.get("confidence", "unknown")
            ).strip().lower()

            confidence = confidence_map.get(
                confidence_code,
                "unknown"
            )

            if min_confidence == "nominal" and confidence not in (
                "nominal",
                "high"
            ):
                continue

            if min_confidence == "high" and confidence != "high":
                continue

            region = assign_region(
                latitude,
                longitude,
                region_geometries
            )

            if region is None:
                unassigned_count += 1
                continue

            try:
                frp = float(row.get("frp") or 0)
            except (ValueError, TypeError):
                frp = 0

            detections.append({
                "latitude": latitude,
                "longitude": longitude,
                "region": region,
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
            "min_confidence": min_confidence,
            "count": len(detections),
            "raw_valid_detections": valid_raw_count,
            "unassigned_detections": unassigned_count,
            "retrieved_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "regional_assignment": (
                "One detection assigned to one polygon region"
            ),
            "boundary_source": "geoBoundaries",
            "boundary_years": boundary_data["metadata"],
            "detections": detections
        }

    except urllib.error.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"NASA FIRMS returned HTTP {error.code}."
        )

    except urllib.error.URLError:
        raise HTTPException(
            status_code=502,
            detail="Unable to connect to NASA FIRMS."
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="NASA FIRMS processing failed. Check server logs."
        )


@app.get("/api/fires/summary")
def fire_summary(
    days: int = Query(default=1, ge=1, le=5),
    satellite: str = Query(default="NOAA21"),
    min_confidence: str = Query(default="low")
):
    result = get_fires(
        days=days,
        satellite=satellite,
        min_confidence=min_confidence
    )

    counts = {
        name: 0
        for name in (
            "Indian Punjab",
            "Haryana",
            "Rajasthan",
            "Madhya Pradesh",
            "Pakistan Punjab",
            "Delhi NCR",
            "Western Uttar Pradesh"
        )
    }

    for detection in result["detections"]:
        counts[detection["region"]] += 1

    return {
        "status": "live",
        "source": "NASA FIRMS",
        "satellite": satellite,
        "period_days": days,
        "min_confidence": min_confidence,
        "total_detections": result["count"],
        "regional_counts": counts,
        "regional_count_check": (
            sum(counts.values()) == result["count"]
        ),
        "unassigned_detections": result[
            "unassigned_detections"
        ],
        "retrieved_at_utc": result["retrieved_at_utc"]
    }
