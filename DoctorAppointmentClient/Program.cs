using DoctorAppointmentClient.Models;
using System.Net.Http.Json;
using System.Net.Http;
using System;
using System.Collections.Generic;
using System.Linq;

namespace DoctorAppointmentClient
{
    class Program 
    {
        private static readonly HttpClient _httpClient = new HttpClient { BaseAddress = new Uri("https://localhost:7124") };

        static async Task Main(string[] args) 
        {
            Console.WriteLine("The app started");

            bool running = true;

            while (running) { 
                DisplayMenu();
                string choice = Console.ReadLine()?.Trim();

                switch (choice) {

                    case "1":
                        await ScheduleAppointmentAsync();
                        break;
                    case "2":
                        await ViewAppointmentsAsync();
                        break;
                    case "3":
                        running = false;
                        Console.WriteLine("The app ended");
                        break;

                }
                Console.WriteLine("\nPress any key to continue");
                Console.ReadKey(true);
                Console.Clear();
            }
        }

        static void DisplayMenu()
        {
            Console.WriteLine("Doctor Appointment Client");
            Console.WriteLine("1. Schedule an appointment");
            Console.WriteLine("2. View appointments (Doctor's View)");
            Console.WriteLine("3. Exit");
            Console.Write("Enter your choice: ");
        }

        static async Task ScheduleAppointmentAsync() 
        {
            Console.WriteLine("\nScadual new Appointment ");

            Console.Write("Enter Patient name: ");
            string? patientName = Console.ReadLine();

            Console.Write("Enter Contact Info (email or phone): ");
            string? contactInfo = Console.ReadLine();

            Console.Write("Enter Appointment date and time (YYYY-MM-DD HH:MM): ");
            string? dateTimeString = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(patientName) || string.IsNullOrWhiteSpace(contactInfo) || string.IsNullOrWhiteSpace(dateTimeString))
            {
                Console.WriteLine("All fields are required. Please try again.");
                return;
            }

            if (!DateTime.TryParse(dateTimeString, out DateTime appointmentDateTime))
            {
                Console.WriteLine("Invalid date/time format. please use YYYY-MM-DD HH:MM.");
                return;
            }

            var newAppointment = new Appointment
            {
                PatientName = patientName,
                Contactinfo = contactInfo,
                AppointmentDateTime = appointmentDateTime,
                Status = "Scheduled",
                DoctorNote = ""
            };

            try
            {
                var response = await _httpClient.PostAsJsonAsync("api/appointment", newAppointment);
                if (response.IsSuccessStatusCode)
                {
                    var createdAppointment = await response.Content.ReadFromJsonAsync<Appointment>();
                    Console.WriteLine($"Appointment scheduled successfully ID: {createdAppointment?.Id ?? 0}");
                }
                else
                {
                    string errorContent = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"Error scheduling appointment: {response.StatusCode} - {errorContent}");
                }
            }
            catch (HttpRequestException ex)
            {
                Console.WriteLine($"Network error: Could not connect to API. Is the API running? check the API URL in Program.cs. Error: {ex.Message}");
            }

            catch (Exception ex)
            {
                Console.WriteLine($"An unexpected error occurred: {ex.Message}");
            }
        }
        static async Task ViewAppointmentsAsync()
        {
            Console.WriteLine("Doctor's Appointments");

            try
            {
                var appointments = await _httpClient.GetFromJsonAsync<List<Appointment>>("api/appointments");

                if (appointments != null && appointments.Any())
                {
                    foreach (var appt in appointments.OrderBy(a => a.AppointmentDateTime))
                    {
                        Console.WriteLine($"ID: {appt.Id}, Date: {appt.AppointmentDateTime:yyyy-MM-dd HH:mm}, Patient: {appt.PatientName}, Status: {appt.Status}");
                        if (!string.IsNullOrWhiteSpace(appt.DoctorNote))
                        {
                            Console.WriteLine($"  Notes: {appt.DoctorNote}");
                        }
                        Console.WriteLine("-----------------------------------");
                    }
                }
                else
                {
                    Console.WriteLine("No appointments found");
                }
            }
            catch (HttpRequestException ex)
            {
                Console.WriteLine($"Network error: Could not retrieve appointments from API. Is the API running? Please check the API URL in Program.cs. Error: {ex.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An unexpected error occurred: {ex.Message}");
            }
            Console.WriteLine("---------------------------\n");
        }
    }
}