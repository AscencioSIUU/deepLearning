#!/bin/sh
# Genera el reporte en .docx (nativo de Word, editable) y .pdf (render con estilo).
# reference.docx = plantilla de pandoc con estilo de tabla en cuadrícula (ver git log para el parche).
set -e
cd "$(dirname "$0")"
pandoc reporte.md --reference-doc reference.docx -o reporte.docx
pandoc reporte.md --pdf-engine=weasyprint -c report.css -o reporte.pdf
echo "OK -> reporte.docx, reporte.pdf"
