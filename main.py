
import os
import csv
import io
import json
import re
import unicodedata
import urllib.request
import urllib.error
from datetime import datetime, timezone
from threading import Lock

from fastapi import FastAPI, HTTPException, Query
from shapely.geometry import shape, Point
from shapely.ops import unary_union


app = FastAPI(
    title="North India Air Intelligence API",
    description="NASA FIRMS fire monitoring with geographic classification",
    version="2.3.0"
)

NASA_KEY = os.getenv("NASA_FIRMS_MAP_KEY")
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

GEOBOUNDARIES = {
    "india_states": "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/",
    "pakistan_provinces": "https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM1/",
    "india_districts": "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/"
}

SOURCES = {
    "NOAA20": "VIIRS_NOAA20_NRT",
    "NOAA21": "VIIRS_NOAA21_NRT",
    "SNPP": "VIIRS_SNPP_NRT"
}

# Custom Western UP monitoring region.
# This is not an official administrative region.
WEST_UP_DISTRICTS = {
    "Saharanpur", "Muzaffarnagar", "Shamli", "Meerut",
    "Baghpat", "Ghaziabad", "Gautam Buddha Nagar", "Hapur",
    "Bulandshahr", "Aligarh", "Hathras", "Mathura", "Agra",
    "Firozabad", "Etah", "Kasganj", "Moradabad", "Amroha",
    "Bijnor", "Rampur", "Sambhal", "Bareilly", "Pilibhit",
    "Shahjahanpur", "Budaun"
}

# Historical district names mapped to current names.
DISTRICT_ALIASES = {
    "jyotiba phule nagar": "amroha",
    "jyotiba phule nagar district": "amroha",
    "mahamaya nagar": "hathras",
    "kanshiram nagar": "kasganj",
    "kanshi ram nagar": "kasganj",
    "prabuddha nagar": "shamli",
    "prabudh nagar": "shamli"
}

BOUNDARY_CACHE = None
BOUNDARY_LOCK = Lock()


def normalize_name(value):
    value = str(value or "").strip().casefold()
    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )

    return re.sub(r"[^a-z0-9]", "", value)


def canonical_district_name(value):
    normalized = normalize_name(value)

    aliases = {
        normalize_name(old_name): normalize_name(current_name)
        for old_name, current_name in DISTRICT_ALIASES.items()
    }

    return aliases.get(normalized, normalized)


def get_feature_name(properties):
    for key in (
        "shapeName",
        "NAME_1",
        "NAME_2",
        "name",
        "NAME",
        "admin"
    ):
        value = properties.get(key)

        if value:
            return str(value).strip()

    return ""


def download_json(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "NorthIndiaAirIntelligence/2.3"
        }
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_boundary(metadata_url):
    metadata = download_json(metadata_url)

    geojson_url = (
        metadata.get("simplifiedGeometryGeoJSON")
        or metadata.get("gjDownloadURL")
    )

    if not geojson_url:
        raise RuntimeError(
            "geoBoundaries download URL is missing."
        )

    geojson = download_json(geojson_url)

    if not geojson.get("features"):
        raise RuntimeError(
            "Boundary GeoJSON contains no features."
        )

    return metadata, geojson


def load_region_geometries():
    global BOUNDARY_CACHE

    if BOUNDARY_CACHE is not None:
        return BOUNDARY_CACHE

    with BOUNDARY_LOCK:

        if BOUNDARY_CACHE is not None:
            return BOUNDARY_CACHE

        try:
            india_meta, india_data = download_boundary(
                GEOBOUNDARIES["india_states"]
            )

            pakistan_meta, pakistan_data = download_boundary(
                GEOBOUNDARIES["pakistan_provinces"]
            )

            districts_meta, districts_data = download_boundary(
                GEOBOUNDARIES["india_districts"]
            )

            regions = {}
            state_geometries = {}
            available_state_names = []

            # 1. Load Indian state and union territory boundaries.
            for feature in india_data["features"]:
                properties = feature.get("properties", {})
                state_name = get_feature_name(properties)

                if not state_name or not feature.get("geometry"):
                    continue

                normalized = normalize_name(state_name)

                state_geometries[normalized] = shape(
                    feature["geometry"]
                )

                available_state_names.append(state_name)

            state_mapping = {
                "punjab": "Indian Punjab",
                "haryana": "Haryana",
                "rajasthan": "Rajasthan",
                "madhyapradesh": "Madhya Pradesh",
                "nctofdelhi": "Delhi NCR",
                "delhi": "Delhi NCR"
            }

            for state_key, region_name in state_mapping.items():
                geometry = state_geometries.get(
                    normalize_name(state_key)
                )

                if geometry is not None:
                    regions[region_name] = geometry

            # 2. Load Pakistan Punjab.
            for feature in pakistan_data["features"]:
                properties = feature.get("properties", {})
                province_name = get_feature_name(properties)

                if normalize_name(province_name) == "punjab":
                    regions["Pakistan Punjab"] = shape(
                        feature["geometry"]
                    )
                    break

            # 3. Find Uttar Pradesh state polygon.
            up_geometry = state_geometries.get(
                normalize_name("Uttar Pradesh")
            )

            if up_geometry is None:
                raise RuntimeError(
                    "Uttar Pradesh state boundary was not found. "
                    "Available state names: "
                    + ", ".join(sorted(available_state_names))
                )

            # 4. Build Western UP from selected district polygons.
            requested_names = {
                normalize_name(name): name
                for name in WEST_UP_DISTRICTS
            }

            matched_districts = {}
            missing_districts = set(WEST_UP_DISTRICTS)
            observed_candidate_names = set()

            for feature in districts_data["features"]:
                properties = feature.get("properties", {})
                district_name = get_feature_name(properties)

                if not district_name or not feature.get("geometry"):
                    continue

                canonical_name = canonical_district_name(
                    district_name
                )

                if canonical_name not in requested_names:
                    continue

                district_geometry = shape(feature["geometry"])

                # Clip to UP boundary to avoid matching another state.
                clipped_geometry = district_geometry.intersection(
                    up_geometry
                )

                if clipped_geometry.is_empty:
                    continue

                if clipped_geometry.area <= 0:
                    continue

                requested_name = requested_names[canonical_name]

                matched_districts[requested_name] = clipped_geometry
                missing_districts.discard(requested_name)

                observed_candidate_names.add(district_name)

            if matched_districts:
                regions["Western Uttar Pradesh"] = unary_union(
                    list(matched_districts.values())
                )

            # 5. Verify all required regions.
            required_regions = {
                "Indian Punjab",
                "Haryana",
                "Rajasthan",
                "Madhya Pradesh",
                "Delhi NCR",
                "Pakistan Punjab",
                "Western Uttar Pradesh"
            }

            missing_regions = required_regions - set(regions)

            if missing_regions:
                raise RuntimeError(
                    "Missing regional boundaries: "
                    + ", ".join(sorted(missing_regions))
                    + ". Available normalized state names: "
                    + ", ".join(
                        sorted(
                            normalize_name(name)
                            for name in available_state_names
                        )
                    )
                    + ". Available original state names: "
                    + ", ".join(sorted(available_state_names))
                    + ". Matched Western UP districts: "
                    + ", ".join(sorted(matched_districts))
                    + ". Missing Western UP districts: "
                    + ", ".join(sorted(missing_districts))
                )

            BOUNDARY_CACHE = {
                "regions": regions,
                "matched_western_up_districts": sorted(
                    matched_districts.keys()
                ),
                "missing_western_up_districts": sorted(
                    missing_districts
                ),
                "observed_western_up_names": sorted(
                    observed_candidate_names
                ),
                "metadata": {
                    "source": "geoBoundaries",
                    "india_states_year": india_meta.get(
                        "boundaryYearRepresented"
                    ),
                    "pakistan_provinces_year": pakistan_meta.get(
                        "boundaryYearRepresented"
                    ),
                    "india_districts_year": districts_meta.get(
                        "boundaryYearRepresented"
                    )
                }
            }

            return BOUNDARY_CACHE

        except Exception as error:
            raise RuntimeError(
                f"Boundary loading failed: {error}"
            )


def assign_region(latitude, longitude, geometries):
    point = Point(longitude, latitude)

    # Deterministic priority for shared boundary points.
    priority = [
        "Delhi NCR",
        "Indian Punjab",
        "Haryana",
        "Rajasthan",
        "Madhya Pradesh",
        "Western Uttar Pradesh",
        "Pakistan Punjab"
    ]

    for name in priority:
        geometry = geometries.get(name)

        if geometry is not None and geometry.covers(point):
            return name

    return None


@app.get("/")
def root():
    return {
        "message": "North India Air Intelligence API",
        "status": "operational",
        "version": "2.3.0",
        "documentation": "/docs",
        "live_data": "/api/fires"
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
        data = load_region_geometries()

        return {
            "status": "success",
            "source": "geoBoundaries",
            "boundary_years": data["metadata"],
            "regions": [
                {
                    "name": name,
                    "type": (
                        "Custom district union"
                        if name == "Western Uttar Pradesh"
                        else "Administrative boundary"
                    )
                }
                for name in data["regions"]
            ],
            "western_up": {
                "expected_district_count": len(WEST_UP_DISTRICTS),
                "matched_district_count": len(
                    data["matched_western_up_districts"]
                ),
                "missing_district_count": len(
                    data["missing_western_up_districts"]
                ),
                "matched_districts": data[
                    "matched_western_up_districts"
                ],
                "missing_districts": data[
                    "missing_western_up_districts"
                ],
                "observed_dataset_names": data[
                    "observed_western_up_names"
                ]
            }
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
        geometries = boundary_data["regions"]

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
                "User-Agent": "NorthIndiaAirIntelligence/2.3"
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
        raw_count = 0
        unassigned_count = 0
        filtered_count = 0

        for row in reader:
            try:
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
            except (ValueError, TypeError, KeyError):
                continue

            raw_count += 1

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
                filtered_count += 1
                continue

            if min_confidence == "high" and confidence != "high":
                filtered_count += 1
                continue

            region = assign_region(
                latitude,
                longitude,
                geometries
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
            "raw_valid_detections": raw_count,
            "unassigned_detections": unassigned_count,
            "filtered_confidence_detections": filtered_count,
            "retrieved_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "regional_assignment": "One polygon region per detection",
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
