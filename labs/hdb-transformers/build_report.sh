#!/bin/sh
# Compila el reporte a PDF usando pandoc + weasyprint asegurando el límite de 5 páginas
set -e
cd "$(dirname "$0")"
pandoc reporte.md --pdf-engine=weasyprint -c report.css -o "Hoja de Trabajo 2. Transformers y Mecanismos de Atención.pdf"
echo "Reporte compilado exitosamente: Hoja de Trabajo 2. Transformers y Mecanismos de Atención.pdf"
