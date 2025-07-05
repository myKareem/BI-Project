namespace DoctorAppointmentClient.Models
{
    public class Appointment
    {
        public int Id { get; set; } //primary key
        public DateTime AppointmentDateTime { get; set; }
        public string PatientName { get; set; } = string.Empty;
        public string Contactinfo { get; set; } = string.Empty;
        public string DoctorNote { get; set; } = string.Empty;
        public string Status { get; set; } = "Scheduled"; // Scheduled, Completed, Cancelled
    }

    public class Patient
    { 
        public int Id { get; set; } //primary key
        public string name { get; set; } = string.Empty;
        public string contactinfo { get; set; } = string.Empty;
    }
}


