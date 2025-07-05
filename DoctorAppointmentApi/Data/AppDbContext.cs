using Microsoft.EntityFrameworkCore;
using DoctorAppointmentApi.Models;


namespace DoctorAppointmentApi.Data
{
    public class AppDbContext : DbContext {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }
        public DbSet<Appointment> Appointments { get; set; }
    }
}