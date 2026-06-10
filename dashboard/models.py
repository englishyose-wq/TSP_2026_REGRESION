from django.db import models


class LatestSnapshot(models.Model):
    """Último Excel subido y última gráfica generada (singleton, pk=1)."""

    uploaded_file = models.BinaryField(null=True, blank=True)
    uploaded_file_name = models.CharField(max_length=255, blank=True, default="")
    uploaded_sheet = models.CharField(max_length=255, blank=True, default="")
    plot_html = models.TextField(blank=True, default="")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "último estado guardado"
        verbose_name_plural = "último estado guardado"

    def __str__(self):
        return self.uploaded_file_name or "sin archivo"
