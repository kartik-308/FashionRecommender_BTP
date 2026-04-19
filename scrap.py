"""
eBay Fashion Dataset Scraper — Browse API (OAuth2 Client Credentials)
Uses the modern eBay Browse API instead of the deprecated Finding API.
"""

import requests
import pandas as pd
import json
import os
import time
from pathlib import Path
import base64

# ─── CONFIG ──────────────────────────────────────────────────────────────────

APP_ID   = os.environ.get("EBAY_APP_ID")
CERT_ID  = os.environ.get("EBAY_CERT_ID")   # ← paste your Cert ID

OUTPUT_DIR   = Path("ebay_fashion_dataset")
IMAGE_DIR    = OUTPUT_DIR / "images"
ANNOT_DIR    = OUTPUT_DIR / "annotations"
CATALOG_FILE = OUTPUT_DIR / "category_catalog.json"
TOKEN_FILE   = OUTPUT_DIR / ".token_cache.json"   # cached so we don't re-fetch every run

SEARCH_QUERIES = [
    # ── Streetwear & Urban ─────────────────────────────────────────────────
    "graphic tee streetwear",
    "boxy fit tshirt",
    "oversized tshirt men",
    "longline tshirt",
    "tie dye shirt",
    "vintage band tee",
    "drop shoulder tshirt",
    "mesh top streetwear",
    "basketball jersey",
    "football jersey fashion",
    "rugby shirt",
    "skate tshirt",

    # ── Shirts & Formal ────────────────────────────────────────────────────
    "linen shirt men",
    "flannel shirt",
    "denim shirt",
    "hawaiian shirt",
    "western cowboy shirt",
    "mandarin collar shirt",
    "oxford button down shirt",
    "mens dress shirt slim fit",
    "cuban collar shirt",
    "resort shirt mens",

    # ── Jackets & Coats ────────────────────────────────────────────────────
    "varsity jacket",
    "windbreaker jacket",
    "anorak jacket",
    "field jacket",
    "safari jacket",
    "harrington jacket",
    "trucker denim jacket",
    "fleece jacket",
    "sherpa jacket",
    "wax jacket",
    "rain mac jacket",
    "double breasted coat womens",
    "teddy bear coat",
    "cape coat womens",
    "fur coat faux womens",
    "kimono jacket",
    "haori jacket",

    # ── Knitwear ───────────────────────────────────────────────────────────
    "fisherman knit sweater",
    "turtleneck sweater",
    "rollneck jumper",
    "polo neck sweater",
    "argyle sweater vest",
    "cable knit cardigan",
    "mohair sweater",
    "chunky knit jumper",
    "fair isle sweater",

    # ── Bottoms ────────────────────────────────────────────────────────────
    "barrel leg jeans",
    "straight leg jeans",
    "baggy jeans",
    "mom jeans",
    "flared jeans",
    "carpenter jeans",
    "acid wash jeans",
    "parachute pants",
    "linen trousers",
    "pleated trousers",
    "track pants",
    "sweatpants",
    "swim shorts",
    "board shorts",
    "bermuda shorts",
    "denim shorts",
    "biker shorts",
    "cycling shorts women",
    "tennis skirt",
    "denim mini skirt",
    "pleated midi skirt",
    "satin skirt",
    "wrap skirt",
    "asymmetric skirt",

    # ── Dresses ────────────────────────────────────────────────────────────
    "shirt dress",
    "smock dress",
    "babydoll dress",
    "pinafore dress",
    "sundress women",
    "bodycon dress",
    "crochet dress",
    "lace dress",
    "satin slip dress",
    "denim dress",
    "sweater dress",
    "blazer dress",
    "cut out dress",
    "open back dress",
    "halter neck dress",
    "one shoulder dress",
    "tiered ruffle dress",
    "puff sleeve dress",

    # ── Co-ords & Sets ─────────────────────────────────────────────────────
    "matching set womens",
    "two piece co ord set",
    "lounge set womens",
    "biker shorts set",
    "tracksuit set mens",
    "linen co ord set",
    "knit set womens",
    "crop top skirt set",

    # ── Activewear & Gym ───────────────────────────────────────────────────
    "sports bra",
    "gym leggings",
    "yoga pants",
    "compression tights",
    "running shorts mens",
    "athletic tank top",
    "gym hoodie",
    "training jacket",
    "cycling jersey",
    "tennis dress",
    "golf polo shirt",
    "rashguard swim top",

    # ── Swimwear ───────────────────────────────────────────────────────────
    "bikini top",
    "bikini bottom",
    "one piece swimsuit",
    "tankini swimsuit",
    "swim trunks mens",
    "rash vest",

    # ── Loungewear & Sleepwear ─────────────────────────────────────────────
    "pajama set womens",
    "silk robe",
    "satin nightdress",
    "mens lounge pants",
    "women oversized sleep shirt",

    # ── Footwear ───────────────────────────────────────────────────────────
    "mary jane shoes",
    "ballet flats",
    "kitten heels",
    "block heel sandals",
    "strappy heels",
    "espadrilles",
    "wedge sandals",
    "dad sneakers",
    "high top sneakers",
    "slip on sneakers",
    "boat shoes mens",
    "monk strap shoes",
    "derby shoes",
    "oxford shoes",
    "knee high boots",
    "ankle boots",
    "cowboy boots",
    "ugg style boots",
    "rain boots",
    "hiking boots",
    "flip flops",
    "slider sandals",
    "birkenstock style sandals",

    # ── Bags ───────────────────────────────────────────────────────────────
    "shoulder bag leather",
    "clutch bag",
    "belt bag fanny pack",
    "backpack fashion",
    "gym duffel bag",
    "woven bag",
    "straw bag",
    "doctor bag",
    "top handle bag",
    "chain strap bag",
    "mesh bag",

    # ── Headwear ───────────────────────────────────────────────────────────
    "fedora hat",
    "wide brim hat",
    "cowboy hat",
    "trucker hat",
    "knit beanie",
    "balaclava",
    "headband women",
    "hair scarf",

    # ── Scarves, Belts & Jewellery ─────────────────────────────────────────
    "silk scarf",
    "mens leather belt",
    "chain belt women",
    "chunky necklace",
    "layered necklace",
    "hoop earrings",
    "statement earrings",
    "cuff bracelet",
    "charm bracelet",
    "rings set women",
    "signet ring mens",

    # ── Sunglasses ─────────────────────────────────────────────────────────
    "cat eye sunglasses",
    "oversized sunglasses",
    "retro round sunglasses",
    "sport sunglasses",
    "shield sunglasses",

    # ── Socks & Tights ─────────────────────────────────────────────────────
    "knee high socks",
    "fishnet tights",
    "sheer tights",
    "ankle socks pack",
    "compression socks",

    # ── Gloves & Scarves ───────────────────────────────────────────────────
    "leather gloves",
    "knit gloves",
    "infinity scarf",
    "blanket scarf plaid",

    # ── Workwear & Smart ───────────────────────────────────────────────────
    "womens suit blazer",
    "mens suit jacket",
    "waistcoat vest mens",
    "tailored trousers womens",
    "pencil skirt",
    "work blouse womens",

    # ── Modest & Cultural Fashion ──────────────────────────────────────────
    "abaya fashion",
    "modest maxi dress",
    "hijab scarf",
    "kurta mens",
    "salwar kameez womens",
    "kaftan dress",

    # ── Vintage & Retro ────────────────────────────────────────────────────
    "70s style flare pants",
    "retro windbreaker",
    "vintage denim jacket",
    "80s style tracksuit",
    "retro polo shirt",
    "vintage corduroy jacket",
    "90s style baggy jeans",
]

PAGES_PER_QUERY = 3      # each page = 50 items → 150 items per keyword
ITEMS_PER_PAGE  = 50     # Browse API max per page is 200, 50 is safe

# ─── OAUTH TOKEN ─────────────────────────────────────────────────────────────

def get_oauth_token() -> str:
    """
    Get Application token via Client Credentials Grant.
    Caches to disk so we don't re-fetch until it expires.
    """
    # Check cache first
    if TOKEN_FILE.exists():
        cached = json.loads(TOKEN_FILE.read_text())
        if time.time() < cached.get("expires_at", 0) - 60:   # 60s buffer
            print("  Using cached OAuth token")
            return cached["access_token"]

    print("  Fetching new OAuth token...")

    credentials = base64.b64encode(f"{APP_ID}:{CERT_ID}".encode()).decode()

    resp = requests.post(
        "https://api.ebay.com/identity/v1/oauth2/token",
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type":  "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "client_credentials",
            "scope":      "https://api.ebay.com/oauth/api_scope",
        },
        timeout=15,
    )

    if resp.status_code != 200:
        print(f"  ✗ Token fetch failed: {resp.status_code} — {resp.text}")
        return ""

    data         = resp.json()
    access_token = data.get("access_token", "")
    expires_in   = data.get("expires_in", 7200)

    # Cache it
    TOKEN_FILE.write_text(json.dumps({
        "access_token": access_token,
        "expires_at":   time.time() + expires_in,
    }))

    print(f"  ✓ Token obtained (expires in {expires_in//60} mins)")
    return access_token


# ─── BROWSE API SEARCH ────────────────────────────────────────────────────────

def search_ebay(keyword: str, token: str, offset: int = 0) -> tuple[list, int]:
    """
    Search eBay Browse API. Returns (items_list, total_count).
    offset = page * ITEMS_PER_PAGE
    """
    headers = {
        "Authorization":              f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID":    "EBAY_US",
        "X-EBAY-C-ENDUSERCTX":        "contextualLocation=country%3DUS",
        "Content-Type":               "application/json",
    }

    params = {
        "q":               keyword,
        "category_ids":    "11450",          # Clothing, Shoes & Accessories
        "filter":          "conditions:{NEW},buyingOptions:{FIXED_PRICE}",
        "fieldgroups":     "EXTENDED",       # includes description & more image fields
        "limit":           str(ITEMS_PER_PAGE),
        "offset":          str(offset),
    }

    try:
        resp = requests.get(
            "https://api.ebay.com/buy/browse/v1/item_summary/search",
            headers=headers,
            params=params,
            timeout=15,
        )

        if resp.status_code == 401:
            print("    ✗ Token expired mid-run")
            return [], 0

        if resp.status_code != 200:
            print(f"    ✗ API error {resp.status_code}: {resp.text[:200]}")
            return [], 0

        data  = resp.json()
        items = data.get("itemSummaries", [])
        total = data.get("total", 0)
        return items, total

    except requests.exceptions.RequestException as e:
        print(f"    ✗ Network error: {e}")
        return [], 0


def extract_item_fields(raw: dict, keyword: str) -> dict:
    """Flatten Browse API item into simple dict."""
    # Browse API gives richer image data
    image_url = ""
    img = raw.get("image", {})
    if img:
        image_url = img.get("imageUrl", "")

    # Additional images if primary missing
    if not image_url:
        additional = raw.get("additionalImages", [])
        if additional:
            image_url = additional[0].get("imageUrl", "")

    # Price
    price_obj = raw.get("price", {})
    price     = price_obj.get("value", "")
    currency  = price_obj.get("currency", "")

    # Category — Browse API returns a list
    categories = raw.get("categories", [])
    category   = categories[0].get("categoryName", "unknown") if categories else "unknown"

    return {
        "item_id":    raw.get("itemId", ""),
        "title":      raw.get("title", ""),
        "image_url":  image_url,
        "category":   category,
        "price":      price,
        "currency":   currency,
        "condition":  raw.get("condition", ""),
        "keyword":    keyword,
        "item_url":   raw.get("itemWebUrl", ""),
        "seller":     raw.get("seller", {}).get("username", ""),
    }


# ─── CATEGORY REGISTRY ───────────────────────────────────────────────────────

class CategoryRegistry:
    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            data            = json.loads(path.read_text())
            self.name_to_id = data["name_to_id"]
            self.id_to_name = {int(k): v for k, v in data["id_to_name"].items()}
            self._next_id   = data["next_id"]
        else:
            self.name_to_id = {}
            self.id_to_name = {}
            self._next_id   = 1

    def get_or_create(self, name: str) -> int:
        key = (name or "unknown").strip()
        if key not in self.name_to_id:
            self.name_to_id[key]           = self._next_id
            self.id_to_name[self._next_id] = key
            self._next_id += 1
            self.save()
        return self.name_to_id[key]

    def save(self):
        self.path.write_text(json.dumps({
            "name_to_id": self.name_to_id,
            "id_to_name": self.id_to_name,
            "next_id":    self._next_id,
        }, indent=2))

    def summary(self):
        print(f"\n  Category catalog — {len(self.name_to_id)} unique categories:")
        for name, cid in sorted(self.name_to_id.items(), key=lambda x: x[1]):
            print(f"    [{cid:04d}] {name}")


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def download_image(url: str, dest: Path) -> bool:
    if not url:
        return False
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
            dest.write_bytes(r.content)
            return True
    except Exception as e:
        print(f"    ✗ Image error: {e}")
    return False


def build_annotation(item: dict, six_digit: str, category_id: int) -> dict:
    return {
        "source":         "shop",
        "pair_id":        int(six_digit),
        "ebay_item_id":   item["item_id"],
        "ebay_item_url":  item["item_url"],
        "search_keyword": item["keyword"],
        "item_1": {
            "category_name": item["category"],
            "category_id":   category_id,
            "style":         1,
            "title":         item["title"],
            "condition":     item["condition"],
            "price":         item["price"],
            "currency":      item["currency"],
            "image_url":     item["image_url"],
            "bounding_box":  None,
            "landmarks":     None,
            "segmentation":  None,
            "scale":         2,
            "occlusion":     1,
            "zoom_in":       1,
            "viewpoint":     2,
        }
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

def fetch_and_save():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    # Get OAuth token
    print("Authenticating with eBay...")
    token = get_oauth_token()
    if not token:
        print("Cannot proceed without a valid token. Check your APP_ID and CERT_ID.")
        return

    # Quick connection test
    print("\nTesting API connection...")
    test_items, test_total = search_ebay("mens shirt", token, offset=0)
    if not test_items:
        print("✗ API test failed — check credentials")
        return
    print(f"✓ API working — {test_total} total results for test query\n")

    registry     = CategoryRegistry(CATALOG_FILE)
    existing     = list(ANNOT_DIR.glob("*.json"))
    item_counter = len(existing) + 1
    print(f"Resuming from item #{item_counter:06d}  ({len(existing)} already scraped)\n")

    summary_rows = []

    for keyword in SEARCH_QUERIES:
        print(f"\n{'─'*60}")
        print(f"  Keyword: '{keyword}'")
        print(f"{'─'*60}")

        for page in range(PAGES_PER_QUERY):
            offset = page * ITEMS_PER_PAGE
            print(f"  Page {page+1}/{PAGES_PER_QUERY} (offset {offset}) … ", end="", flush=True)

            raw_items, total = search_ebay(keyword, token, offset)

            # Refresh token if it expired
            if not raw_items and total == 0:
                token = get_oauth_token()
                raw_items, total = search_ebay(keyword, token, offset)

            print(f"{len(raw_items)} items  (total available: {total})")

            for raw in raw_items:
                six_digit   = f"{item_counter:06d}"
                item        = extract_item_fields(raw, keyword)
                category_id = registry.get_or_create(item["category"])

                img_path   = IMAGE_DIR / f"{six_digit}.jpg"
                downloaded = download_image(item["image_url"], img_path)

                annotation = build_annotation(item, six_digit, category_id)
                (ANNOT_DIR / f"{six_digit}.json").write_text(
                    json.dumps(annotation, indent=2)
                )

                summary_rows.append({
                    "id":          six_digit,
                    "keyword":     keyword,
                    "item_id":     item["item_id"],
                    "title":       item["title"],
                    "category":    item["category"],
                    "category_id": category_id,
                    "price":       item["price"],
                    "image_url":   item["image_url"],
                    "image_saved": downloaded,
                })

                item_counter += 1

            time.sleep(1)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "dataset_index.csv"
    df_new   = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    if not df_new.empty:
        if csv_path.exists():
            df_old   = pd.read_csv(csv_path)
            df_final = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset="item_id")
        else:
            df_final = df_new

        df_final.to_csv(csv_path, index=False)

        print(f"\n{'='*60}")
        print(f"  Items this run  : {len(df_new)}")
        print(f"  Total in dataset: {len(df_final)}")
        print(f"  Images saved    : {df_new['image_saved'].sum()}/{len(df_new)}")
        print(f"  Output folder   : {OUTPUT_DIR.resolve()}")
        registry.summary()
    else:
        print("\n  No items scraped.")


if __name__ == "__main__":
    fetch_and_save()