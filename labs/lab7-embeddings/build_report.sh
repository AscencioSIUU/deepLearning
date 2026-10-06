#!/bin/sh
# Compila el reporte a PDF usando pandoc + weasyprint asegurando el límite de 5 páginas
set -e
cd "$(dirname "$0")"
pandoc reporte.md --pdf-engine=weasyprint -c report.css -o "Laboratorio 7. NLP end-to-end y Embeddings.pdf"
echo "Reporte compilado exitosamente: Laboratorio 7. NLP end-to-end y Embeddings.pdf"
