# Generated by Django 4.2.5 on 2023-11-27 12:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('timeseries_api', '0002_alter_timeseriesdatasets_input_values'),
    ]

    operations = [
        migrations.CreateModel(
            name='SelectedFeature',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('feature_name', models.CharField(max_length=255)),
            ],
        ),
        migrations.AddField(
            model_name='timeseriesdatasets',
            name='model',
            field=models.BinaryField(null=True),
        ),
    ]
