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
    snapshot.save(
        update_fields=["uploaded_file", "uploaded_file_name", "uploaded_sheet", "updated_at"]
    )


def load_uploaded_file():
    snapshot = LatestSnapshot.objects.filter(pk=SINGLETON_PK).first()
    if not snapshot or not snapshot.uploaded_file or not snapshot.uploaded_file_name:
        return None, None, None
    sheet_name = snapshot.uploaded_sheet or None
    return snapshot.uploaded_file, snapshot.uploaded_file_name, sheet_name


def save_plot_html(plot_html):
    if not plot_html:
        return
    snapshot = _get_snapshot()
    snapshot.plot_html = plot_html
    snapshot.save(update_fields=["plot_html", "updated_at"])


def load_plot_html():
    snapshot = LatestSnapshot.objects.filter(pk=SINGLETON_PK).first()
    if snapshot and snapshot.plot_html:
        return snapshot.plot_html
    return None


def load_latest_upload_metadata():
    snapshot = LatestSnapshot.objects.filter(pk=SINGLETON_PK).first()
    if not snapshot or not snapshot.uploaded_file_name:
        return None, None, None
    sheet_name = snapshot.uploaded_sheet or None
    return snapshot.uploaded_file_name, sheet_name, snapshot.updated_at
