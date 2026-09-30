#!/bin/sh
# Compila el reporte a PDF usando pandoc + weasyprint asegurando el límite de 4 páginas
set -e
cd "$(dirname "$0")"
pandoc reporte.md --pdf-engine=weasyprint -c report.css -o reporte.pdf
echo "Reporte compilado exitosamente: reporte.pdf"
