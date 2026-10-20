# INSIGHTFACE
## `POST /extract-faces`

Returns **every** face in the image with its embedding in a single call (replaces Face++ `/detect` + one `/extract-embedding` per crop).

- Body: `multipart/form-data` with `file`.
- Query (optional): `min_face_size` (px in the original image, default `40`), `min_det_score` (default `0.5`).
- Response: `{ success, faces_found, faces_skipped, image_size, faces: [{ bbox: {x1,y1,x2,y2}, confidence, embedding[512] }] }`.
- `bbox` is in the coordinates of the **original** uploaded image.

Embeddings come from the same `buffalo_sc` model as the other endpoints, so they can be compared directly with existing `reference_embedding`s.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `FACE_API_KEY` | _(unset)_ | When set, every route except `/` requires header `X-API-Key: <value>`; otherwise 401. |
| `MULTI_DET_SIZE` | `1280` | Detector input size for `/extract-faces`. 640 misses small faces in group photos. |
| `MULTI_MAX_SIDE` | `1920` | Image is downscaled to this longest side before detection. |

## Similarity scale

`/compare` and `/search` return raw cosine similarity of `buffalo_sc` embeddings. Same person usually lands around 0.4–0.8 depending on pose/lighting; different people usually below 0.3. `is_match` now uses 0.45 (it used 0.75, which rejected most real matches).
