import os
import csv
import io
import json
import re
import unicodedata
import urllib.request
import urllib.error
from urllib.parse import urlencode
from datetime import datetime, timezone
from threading import Lock

from fastapi import FastAPI, HTTPException, Query
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely import make_valid

APP_VERSION = "2.5.0"

app = FastAPI(
    title="North India Air Intelligence API",
    description="NASA FIRMS fire monitoring with geographic classification",
    version=APP_VERSION
)

NASA_KEY = os.getenv("NASA_FIRMS_MAP_KEY")
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

GEOBOUNDARIES = {
    "india_states": "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/",
    "pakistan_provinces": "https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM1/",
    "india_districts": "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/"
}

ESRI_SHAMLI_QUERY = (
    "https://livingatlas.esri.in/server/rest/services/"
    "IAB2024/India_Administrative_Boundaries_2024/MapServer/2/query"
)

SOURCES = {
    "NOAA20": "VIIRS_NOAA20_NRT",
    "NOAA21": "VIIRS_NOAA21_NRT",
    "SNPP": "VIIRS_SNPP_NRT"
}

WEST_UP_DISTRICTS = {
    "Saharanpur", "Muzaffarnagar", "Shamli", "Meerut",
    "Baghpat", "Ghaziabad", "Gautam Buddha Nagar", "Hapur",
    "Bulandshahr", "Aligarh", "Hathras", "Mathura", "Agra",
    "Firozabad", "Etah", "Kasganj", "Moradabad", "Amroha",
    "Bijnor", "Rampur", "Sambhal", "Bareilly", "Pilibhit",
    "Shahjahanpur", "Budaun"
}

DISTRICT_ALIASES = {
    "jyotiba phule nagar": "amroha",
    "mahamaya nagar": "hathras",
    "kanshiram nagar": "kasganj",
    "kanshi ram nagar": "kasganj",
    "prabuddha nagar": "shamli",
    "prabudh nagar": "shamli",
    "shamli district": "shamli"
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
        normalize_name(old): normalize_name(new)
        for old, new in DISTRICT_ALIASES.items()
    }
    return aliases.get(normalized, normalized)


def get_feature_name(properties):
    for key in ("shapeName", "NAME_1", "NAME_2", "name", "NAME", "admin"):
        value = properties.get(key)
        if value:
            return str(value).strip()
    return ""


def download_json(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"NorthIndiaAirIntelligence/{APP_VERSION}"}
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
        raise RuntimeError("geoBoundaries download URL is missing.")
    geojson = download_json(geojson_url)
    if not geojson.get("features"):
        raise RuntimeError("Boundary GeoJSON contains no features.")
    return metadata, geojson


def safe_geometry(geojson_geometry):
    geometry = shape(geojson_geometry)
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    if geometry.is_empty:
        raise ValueError("Boundary geometry is empty.")
    return geometry


def fetch_esri_shamli(up_geometry):
    """Return a validated Shamli polygon from Esri, or None with diagnostics."""
    params = {
        "where": (
            "state='Uttar Pradesh' AND "
            "(name='Shamli' OR lgd_districtname='Shamli' "
            "OR censusname='Shamli')"
        ),
        "outFields": (
            "name,state,censusname,lgd_districtname,lgd_districtcode"
        ),
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson"
    }
    url = ESRI_SHAMLI_QUERY + "?" + urlencode(params)

    try:
        result = download_json(url)
        features = result.get("features") or []
        if not features:
            return None, {
                "district": "Shamli",
                "source": "Esri India Living Atlas 2024",
                "method": "not_added",
                "reason": "Esri query returned no matching Shamli feature."
            }

        valid_candidates = []
        rejected = []

        for feature in features:
            props = feature.get("properties") or {}
            district_names = [
                props.get("name"),
                props.get("lgd_districtname"),
                props.get("censusname")
            ]
            state_name = props.get("state")

            if normalize_name(state_name) != normalize_name("Uttar Pradesh"):
                rejected.append("Feature state was not Uttar Pradesh.")
                continue

            if not any(
                normalize_name(value) == "shamli"
                for value in district_names if value
            ):
                rejected.append("Feature district name was not Shamli.")
                continue

            geometry_data = feature.get("geometry")
            if not geometry_data:
                rejected.append("Matching feature had no geometry.")
                continue

            try:
                district_geometry = safe_geometry(geometry_data)
                clipped = district_geometry.intersection(up_geometry)
                if clipped.is_empty or clipped.area <= 0:
                    rejected.append("Shamli polygon did not overlap UP boundary.")
                    continue

                # Reject a tiny sliver or an implausibly broad geometry.
                ratio = clipped.area / max(district_geometry.area, 1e-12)
                if ratio < 0.50:
                    rejected.append(
                        "Less than half of the returned polygon overlapped UP."
                    )
                    continue

                point = clipped.representative_point()
                if not up_geometry.covers(point):
                    rejected.append("Representative point fell outside UP.")
                    continue

                valid_candidates.append((clipped, props, ratio))
            except Exception as exc:
                rejected.append(f"Geometry validation failed: {exc}")

        if not valid_candidates:
            return None, {
                "district": "Shamli",
                "source": "Esri India Living Atlas 2024",
                "method": "not_added",
                "reason": "; ".join(rejected) or "No valid candidate."
            }

        # One feature is expected. If duplicates occur, use the largest
        # validated clipped polygon rather than merging potentially duplicated data.
        geometry, props, ratio = max(
            valid_candidates,
            key=lambda item: item[0].area
        )
        return geometry, {
            "district": "Shamli",
            "dataset_name": props.get("name"),
            "state": props.get("state"),
            "lgd_district_name": props.get("lgd_districtname"),
            "lgd_district_code": props.get("lgd_districtcode"),
            "source": "Esri India Living Atlas 2024",
            "method": "validated_esri_polygon_clipped_to_up",
            "geometry_valid": geometry.is_valid,
            "up_overlap_ratio": round(ratio, 4),
            "intersection_area": round(geometry.area, 8)
        }

    except Exception as exc:
        return None, {
            "district": "Shamli",
            "source": "Esri India Living Atlas 2024",
            "method": "not_added",
            "reason": f"Esri fallback unavailable: {exc}"
        }


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

            for feature in india_data["features"]:
                properties = feature.get("properties", {})
                name = get_feature_name(properties)
                if not name or not feature.get("geometry"):
                    continue
                state_geometries[normalize_name(name)] = safe_geometry(
                    feature["geometry"]
                )
                available_state_names.append(name)

            state_mapping = {
                "punjab": "Indian Punjab",
                "haryana": "Haryana",
                "rajasthan": "Rajasthan",
                "madhya pradesh": "Madhya Pradesh",
                "nct of delhi": "Delhi NCR",
                "delhi": "Delhi NCR"
            }

            for state_name, region_name in state_mapping.items():
                geometry = state_geometries.get(normalize_name(state_name))
                if geometry is not None:
                    regions[region_name] = geometry

            for feature in pakistan_data["features"]:
                properties = feature.get("properties", {})
                province_name = get_feature_name(properties)
                if normalize_name(province_name) == "punjab":
                    regions["Pakistan Punjab"] = safe_geometry(
                        feature["geometry"]
                    )
                    break

            up_geometry = state_geometries.get(
                normalize_name("Uttar Pradesh")
            )
            if up_geometry is None:
                raise RuntimeError(
                    "Uttar Pradesh boundary not found. Available states: "
                    + ", ".join(sorted(available_state_names))
                )

            requested_names = {
                normalize_name(name): name for name in WEST_UP_DISTRICTS
            }
            matched = {}
            missing = set(WEST_UP_DISTRICTS)
            diagnostics = []

            for feature in districts_data["features"]:
                properties = feature.get("properties", {})
                district_name = get_feature_name(properties)
                if not district_name or not feature.get("geometry"):
                    continue

                canonical = canonical_district_name(district_name)
                if canonical not in requested_names:
                    continue

                requested_name = requested_names[canonical]
                district_geometry = safe_geometry(feature["geometry"])
                clipped = district_geometry.intersection(up_geometry)

                if not clipped.is_empty and clipped.area > 0:
                    matched[requested_name] = clipped
                    missing.discard(requested_name)
                    diagnostics.append({
                        "district": requested_name,
                        "dataset_name": district_name,
                        "source": "geoBoundaries",
                        "method": "intersection",
                        "geometry_valid": district_geometry.is_valid,
                        "intersection_area": round(clipped.area, 8)
                    })
                    continue

                representative_point = district_geometry.representative_point()
                if up_geometry.covers(representative_point):
                    matched[requested_name] = district_geometry
                    missing.discard(requested_name)
                    diagnostics.append({
                        "district": requested_name,
                        "dataset_name": district_name,
                        "source": "geoBoundaries",
                        "method": "representative_point_fallback",
                        "geometry_valid": district_geometry.is_valid,
                        "original_area": round(district_geometry.area, 8)
                    })
                else:
                    diagnostics.append({
                        "district": requested_name,
                        "dataset_name": district_name,
                        "source": "geoBoundaries",
                        "method": "excluded",
                        "reason": "No valid intersection with UP boundary"
                    })

            # Esri is used only for the missing Shamli district, never to
            # replace the other 24 geoBoundaries district polygons.
            shamli_diagnostic = None
            if "Shamli" in missing:
                shamli_geometry, shamli_diagnostic = fetch_esri_shamli(
                    up_geometry
                )
                if shamli_geometry is not None:
                    matched["Shamli"] = shamli_geometry
                    missing.discard("Shamli")
                    diagnostics.append(shamli_diagnostic)
                else:
                    diagnostics.append(shamli_diagnostic)

            if matched:
                regions["Western Uttar Pradesh"] = unary_union(
                    list(matched.values())
                )

            required_regions = {
                "Indian Punjab", "Haryana", "Rajasthan",
                "Madhya Pradesh", "Delhi NCR", "Pakistan Punjab",
                "Western Uttar Pradesh"
            }
            missing_regions = required_regions - set(regions)
            if missing_regions:
                raise RuntimeError(
                    "Missing regional boundaries: "
                    + ", ".join(sorted(missing_regions))
                    + ". Matched Western UP districts: "
                    + ", ".join(sorted(matched))
                    + ". Missing Western UP districts: "
                    + ", ".join(sorted(missing))
                )

            BOUNDARY_CACHE = {
                "regions": regions,
                "matched_western_up_districts": sorted(matched),
                "missing_western_up_districts": sorted(missing),
                "district_diagnostics": diagnostics,
                "metadata": {
                    "source": "geoBoundaries with Esri fallback for Shamli",
                    "india_states_year": india_meta.get(
                        "boundaryYearRepresented"
                    ),
                    "pakistan_provinces_year": pakistan_meta.get(
                        "boundaryYearRepresented"
                    ),
                    "india_districts_year": districts_meta.get(
                        "boundaryYearRepresented"
                    ),
                    "shamli_fallback_source": "Esri India Living Atlas 2024"
                }
            }
            return BOUNDARY_CACHE

        except Exception as error:
            raise RuntimeError(f"Boundary loading failed: {error}")


def assign_region(latitude, longitude, geometries):
    point = Point(longitude, latitude)
    priority = [
        "Delhi NCR", "Indian Punjab", "Haryana", "Rajasthan",
        "Madhya Pradesh", "Western Uttar Pradesh", "Pakistan Punjab"
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
        "version": APP_VERSION,
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
            "source": data["metadata"]["source"],
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
                "matched_districts": data["matched_western_up_districts"],
                "missing_districts": data["missing_western_up_districts"],
                "districts_complete": (
                    len(data["matched_western_up_districts"])
                    == len(WEST_UP_DISTRICTS)
                ),
                "diagnostics": data["district_diagnostics"]
            }
        }
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error))


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
    url = f"{FIRMS_BASE}/{NASA_KEY}/{source}/{area}/{days}"

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": f"NorthIndiaAirIntelligence/{APP_VERSION}"}
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
                detail="Unexpected NASA FIRMS data format."
            )

        confidence_map = {
            "l": "low", "n": "nominal", "h": "high",
            "low": "low", "nominal": "nominal", "high": "high"
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
            confidence = confidence_map.get(confidence_code, "unknown")

            if min_confidence == "nominal" and confidence not in (
                "nominal", "high"
            ):
                filtered_count += 1
                continue
            if min_confidence == "high" and confidence != "high":
                filtered_count += 1
                continue

            region = assign_region(latitude, longitude, geometries)
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
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "regional_assignment": "One polygon region per detection",
            "boundary_source": boundary_data["metadata"]["source"],
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
        name: 0 for name in (
            "Indian Punjab", "Haryana", "Rajasthan",
            "Madhya Pradesh", "Pakistan Punjab", "Delhi NCR",
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
        "regional_count_check": sum(counts.values()) == result["count"],
        "unassigned_detections": result["unassigned_detections"],
        "retrieved_at_utc": result["retrieved_at_utc"]
    }
