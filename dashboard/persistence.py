from django.utils import timezone

from .models import LatestSnapshot

SINGLETON_PK = 1


def _get_snapshot():
    snapshot, _ = LatestSnapshot.objects.get_or_create(pk=SINGLETON_PK)
    return snapshot


def save_uploaded_file(file_bytes, filename, sheet_name=None):
    snapshot = _get_snapshot()
    snapshot.uploaded_file = file_bytes
    snapshot.uploaded_file_name = filename
    snapshot.uploaded_sheet = sheet_name or ""
    snapshot.updated_at = timezone.now()
    snapshot.save(
        update_fields=["uploaded_file", "uploaded_file_name", "uploaded_sheet", "updated_at"]
    )


def load_uploaded_file():
    snapshot = LatestSnapshot.objects.filter(pk=SINGLETON_PK).first()
    if not snapshot or not snapshot.uploaded_file or not snapshot.uploaded_file_name:
        return None, None, None
    sheet_name = snapshot.uploaded_sheet or None
    return snapshot.uploaded_file, snapshot.uploaded_file_name, sheet_name


def save_plot_snapshot(plot_html, metadata=None, plot_html_embed=None):
    if not plot_html:
        return
    snapshot = _get_snapshot()
    snapshot.plot_html = plot_html
    snapshot.plot_html_embed = plot_html_embed or plot_html
    snapshot.plot_metadata = metadata or {}
    snapshot.updated_at = timezone.now()
    snapshot.save(
        update_fields=["plot_html", "plot_html_embed", "plot_metadata", "updated_at"]
    )


def load_plot_html():
    snapshot = LatestSnapshot.objects.filter(pk=SINGLETON_PK).first()
    if snapshot and snapshot.plot_html:
        return snapshot.plot_html
    return None


def load_plot_html_embed():
    snapshot = LatestSnapshot.objects.filter(pk=SINGLETON_PK).first()
    if snapshot and snapshot.plot_html_embed:
        return snapshot.plot_html_embed
    if snapshot and snapshot.plot_html:
        return snapshot.plot_html
    return None


def load_plot_metadata():
    snapshot = LatestSnapshot.objects.filter(pk=SINGLETON_PK).first()
    if snapshot and snapshot.plot_metadata:
        return snapshot.plot_metadata
    return {}


def load_snapshot_status():
    snapshot = LatestSnapshot.objects.filter(pk=SINGLETON_PK).first()
    if not snapshot:
        return {
            "has_plot": False,
            "has_excel": False,
            "updated_at": None,
            "view_mode": None,
        }
    return {
        "has_plot": bool(snapshot.plot_html),
        "has_excel": bool(snapshot.uploaded_file_name),
        "updated_at": snapshot.updated_at,
        "view_mode": (snapshot.plot_metadata or {}).get("view_mode"),
    }


def load_latest_upload_metadata():
    snapshot = LatestSnapshot.objects.filter(pk=SINGLETON_PK).first()
    if not snapshot or not snapshot.uploaded_file_name:
        return None, None, None
    sheet_name = snapshot.uploaded_sheet or None
    return snapshot.uploaded_file_name, sheet_name, snapshot.updated_at
